-- ============================================================================
-- MIMIC-III cohort for RareSel-DP (BigQuery Standard SQL)
--
-- One row per hospital admission of an adult (>= 18 y) with at least one ICU stay.
-- Target: MORTALITY = in-hospital death (admissions.hospital_expire_flag).
-- Vital signs are aggregated over the FIRST 24 HOURS of the FIRST ICU stay of the
-- admission, so they are available at prediction time (no post-outcome leakage).
-- To aggregate over the whole stay instead, change
-- the window in `vitals` below — and say so in the paper.
--
-- Requires PhysioNet credentialed access to the `physionet-data` BigQuery project.
-- Billing goes to YOUR project. Scans roughly 10-30 GB (mostly chartevents).
-- Expected size: about 50,000 rows, about 11-12% positive.
-- ============================================================================

WITH adm AS (
  SELECT
    a.subject_id, a.hadm_id, a.admittime, a.dischtime,
    a.admission_type, a.insurance, a.marital_status,
    a.hospital_expire_flag,
    p.gender,
    -- MIMIC-III shifts ages > 89 to ~300 years; cap at 90
    LEAST(DATETIME_DIFF(a.admittime, p.dob, DAY) / 365.242, 90) AS age
  FROM `physionet-data.mimiciii_clinical.admissions` a
  JOIN `physionet-data.mimiciii_clinical.patients` p USING (subject_id)
),
icu AS (
  SELECT
    hadm_id,
    COUNT(*) AS icu_stays,
    SUM(los) AS icu_los,
    ARRAY_AGG(STRUCT(icustay_id, intime, first_careunit) ORDER BY intime LIMIT 1)[OFFSET(0)] AS first_stay
  FROM `physionet-data.mimiciii_clinical.icustays`
  GROUP BY hadm_id
),
dx AS (
  SELECT hadm_id, COUNT(*) AS num_diagnoses
  FROM `physionet-data.mimiciii_clinical.diagnoses_icd` GROUP BY hadm_id
),
px AS (
  SELECT hadm_id, COUNT(*) AS num_procedures
  FROM `physionet-data.mimiciii_clinical.procedures_icd` GROUP BY hadm_id
),
vitals AS (
  SELECT
    i.hadm_id,
    CASE
      WHEN ce.itemid IN (211, 220045) AND ce.valuenum > 0 AND ce.valuenum < 300 THEN 'HR'
      WHEN ce.itemid IN (646, 220277) AND ce.valuenum > 0 AND ce.valuenum <= 100 THEN 'SPO2'
      WHEN ce.itemid IN (223761, 678) AND ce.valuenum > 70 AND ce.valuenum < 120 THEN 'TEMPF'
      WHEN ce.itemid IN (223762, 676) AND ce.valuenum > 10 AND ce.valuenum < 50 THEN 'TEMPC'
    END AS vital,
    ce.valuenum
  FROM `physionet-data.mimiciii_clinical.chartevents` ce
  JOIN icu i ON ce.icustay_id = i.first_stay.icustay_id
  WHERE ce.itemid IN (211, 220045, 646, 220277, 223761, 678, 223762, 676)
    AND ce.valuenum IS NOT NULL
    AND (ce.error IS NULL OR ce.error = 0)
    AND ce.charttime BETWEEN i.first_stay.intime AND DATETIME_ADD(i.first_stay.intime, INTERVAL 24 HOUR)
),
vit AS (
  SELECT
    hadm_id, vital,
    -- convert Fahrenheit to Celsius so both temperature item families agree
    IF(vital = 'TEMPF', (valuenum - 32) / 1.8, valuenum) AS v
  FROM vitals WHERE vital IS NOT NULL
),
vagg AS (
  SELECT
    hadm_id,
    MIN(IF(vital = 'HR', v, NULL))  AS HR_MIN,  AVG(IF(vital = 'HR', v, NULL))  AS HR_MEAN,  MAX(IF(vital = 'HR', v, NULL))  AS HR_MAX,
    MIN(IF(vital = 'SPO2', v, NULL)) AS SPO2_MIN, AVG(IF(vital = 'SPO2', v, NULL)) AS SPO2_MEAN, MAX(IF(vital = 'SPO2', v, NULL)) AS SPO2_MAX,
    MIN(IF(vital IN ('TEMPF','TEMPC'), v, NULL)) AS TEMP_MIN,
    AVG(IF(vital IN ('TEMPF','TEMPC'), v, NULL)) AS TEMP_MEAN,
    MAX(IF(vital IN ('TEMPF','TEMPC'), v, NULL)) AS TEMP_MAX
  FROM vit GROUP BY hadm_id
)
SELECT
  adm.subject_id                          AS SUBJECT_ID,   -- dropped at load (id_cols)
  adm.hadm_id                             AS HADM_ID,      -- dropped at load (id_cols)
  ROUND(adm.age, 1)                       AS AGE,
  adm.gender                              AS GENDER,
  adm.admission_type                      AS ADMISSION_TYPE,
  adm.insurance                           AS INSURANCE,
  IFNULL(adm.marital_status, 'UNKNOWN')   AS MARITAL_STATUS,
  icu.first_stay.first_careunit           AS FIRST_CAREUNIT,
  DATETIME_DIFF(adm.dischtime, adm.admittime, HOUR) / 24.0 AS HOSPITAL_LOS,
  icu.icu_los                             AS ICU_LOS,
  icu.icu_stays                           AS ICU_STAYS,
  IFNULL(dx.num_diagnoses, 0)             AS NUM_DIAGNOSES,
  IFNULL(px.num_procedures, 0)            AS NUM_PROCEDURES,
  vagg.HR_MIN, vagg.HR_MEAN, vagg.HR_MAX,
  vagg.SPO2_MIN, vagg.SPO2_MEAN, vagg.SPO2_MAX,
  vagg.TEMP_MIN, vagg.TEMP_MEAN, vagg.TEMP_MAX,
  adm.hospital_expire_flag                AS MORTALITY
FROM adm
JOIN icu USING (hadm_id)
LEFT JOIN dx USING (hadm_id)
LEFT JOIN px USING (hadm_id)
LEFT JOIN vagg USING (hadm_id)
WHERE adm.age >= 18
  AND adm.dischtime > adm.admittime
ORDER BY adm.hadm_id
