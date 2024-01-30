-- // Data overview //
--================================================================================================================================

--Retrieve all records in the table
SELECT * FROM MEDICAL_APPOINTMENT_NO_SHOW.APPOINTMENT_SCHEMA.APPOINTMENT_DATA;
--> i. 14 variables including target
--> ii. Data contains information such as: 
--      - Patient demographics: e.g. (age and gender)
--      - Health characteristics: e.g. (diabetes or hypertension)
--      - Appointment-specific details: e.g. (scheduled and appointment dates, and whether the patient received a reminder SMS)
--      - Target: whether a patient was a no-show or attended their appointment
--> iii. 20% of no show rate based on a total record of 110k medical appointments 

--================================================================================================================================



-- // Data cleaning //
--================================================================================================================================

--Check the count and percentage of missing values in each column. Drop columns which have more than 80% null values
SELECT
    COUNT(CASE WHEN PatientId IS NULL THEN 1 END) AS Missing_PatientId,
    COUNT(CASE WHEN AppointmentID IS NULL THEN 1 END) AS Missing_AppointmentID,
    COUNT(CASE WHEN Gender IS NULL THEN 1 END) AS Missing_Gender,
    COUNT(CASE WHEN ScheduledDay IS NULL THEN 1 END) AS Missing_ScheduledDay,
    COUNT(CASE WHEN AppointmentDay IS NULL THEN 1 END) AS Missing_AppointmentDay,
    COUNT(CASE WHEN Age IS NULL THEN 1 END) AS Missing_Age,
    COUNT(CASE WHEN Neighbourhood IS NULL THEN 1 END) AS Missing_Neighbourhood,
    COUNT(CASE WHEN Scholarship IS NULL THEN 1 END) AS Missing_Scholarship,
    COUNT(CASE WHEN Hipertension IS NULL THEN 1 END) AS Missing_Hipertension,
    COUNT(CASE WHEN Diabetes IS NULL THEN 1 END) AS Missing_Diabetes,
    COUNT(CASE WHEN Alcoholism IS NULL THEN 1 END) AS Missing_Alcoholism,
    COUNT(CASE WHEN Handcap IS NULL THEN 1 END) AS Missing_Handcap,
    COUNT(CASE WHEN SMS_received IS NULL THEN 1 END) AS Missing_SMS_received,
    COUNT(CASE WHEN No_show IS NULL THEN 1 END) AS Missing_No_show
FROM
    APPOINTMENT_DATA;
--> No Nulls, so not dropping any columns

--================================================================================================================================



-- //Hypothesis based EDA //
-- 1. Do Males tend to miss more appointments than Females?
-- 2. Is there is any relationship between the scheduled and appointment day difference, and the no-show of patients?
-- 3. Is the no-show common among adult patients aged between 18-30? If not, which age group people have higher no-shows?
--================================================================================================================================

-- 1. Do Males tend to miss more appointments than Females? 
SELECT 
    Gender,
    COUNT(*) AS Total_Appointments,
    SUM(CASE WHEN No_show = 'Yes' THEN 1 ELSE 0 END) AS Missed_Appointments,
    ROUND((SUM(CASE WHEN No_show = 'Yes' THEN 1 ELSE 0 END) / COUNT(*) * 100),1) AS Percentage_Missed
FROM 
    APPOINTMENT_DATA
GROUP BY 
    Gender;
--> Both males and females have a similar rate of missing appointments, around 20%.


-- 2. Is there is any relationship between the scheduled and appointment day difference, and the no-show of patients? 
WITH TimeDifference AS (
    SELECT 
        DATEDIFF(day, ScheduledDay, AppointmentDay) AS DaysDifference,
        No_show
    FROM 
        APPOINTMENT_DATA
)
SELECT 
    DaysDifference,
    COUNT(*) AS Total_Appointments,
    SUM(CASE WHEN No_show = 'Yes' THEN 1 ELSE 0 END) AS Missed_Appointments,
    ROUND((SUM(CASE WHEN No_show = 'Yes' THEN 1 ELSE 0 END) / COUNT(*) * 100),1) AS Percentage_Missed
FROM 
    TimeDifference
GROUP BY 
    DaysDifference
ORDER BY 
    DaysDifference;
--> i. There are 5 appointments with negative days difference which they all have 100% missed appointment rate because they were scheduled for past date. This could be due to data entry errors.
--> ii. A significant number of appointments (38,563) are scheduled on the same day, with a relatively low missed appointment rate of 4.6%.
--> iii. For time differences between 1 day to 80 days, the missed appointment rate fluctuates but generally stays within the range of around 20% to 40%.
--> iv. The longer time differences (>80 days) have varying missed appointment rates, like on: 
--      - day 83 (12.5%), 86 (16.7%)
--      - day 103 (60%), day 104 (75%)   
--      - day 112, 115, 117, 119, 122 (0%)
--      - day 132, 139, 146, 151 (100%)


-- 3. Is the no-show common among adult patients aged between 18-30? If not, which age group people have higher no-shows?
WITH AgeGroups AS (
    SELECT 
        CASE 
            WHEN Age BETWEEN 0 AND 12 THEN '0-12'
            WHEN Age BETWEEN 13 AND 17 THEN '13-17'
            WHEN Age BETWEEN 18 AND 30 THEN '18-30'
            WHEN Age BETWEEN 31 AND 50 THEN '31-50'
            ELSE '50+'
        END AS AgeGroup,
        No_show
    FROM 
        APPOINTMENT_DATA
)
SELECT 
    AgeGroup,
    COUNT(*) AS Total_Appointments,
    SUM(CASE WHEN No_show = 'Yes' THEN 1 ELSE 0 END) AS Missed_Appointments,
    ROUND((SUM(CASE WHEN No_show = 'Yes' THEN 1 ELSE 0 END) / COUNT(*) * 100),1) AS Percentage_Missed
FROM 
    AgeGroups
GROUP BY 
    AgeGroup
ORDER BY 
    AgeGroup;
--> i. The age group of 13-17 years has the highest no-show rate at 26.6% (even they have least appointments scheduled), followed by age group of 18-30 with 24.6%  missed appointment rate.
--> ii. In contrast, the age group 50+ years has the lowest no-show rate at 16.2% (even they have highest number of appointments scheduled).

--================================================================================================================================