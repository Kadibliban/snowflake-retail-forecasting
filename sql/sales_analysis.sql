USE DATABASE RETAIL_ML;
USE SCHEMA ANALYTICS;
USE WAREHOUSE RETAIL_ML_WH;

-- Data review
SELECT
    MIN(SALE_DATE) AS first_sale_date,
    MAX(SALE_DATE) AS last_sale_date,
    COUNT(*) AS rows,
    AVG(REVENUE) AS avg_revenue,
    SUM(REVENUE) AS total_revenue
FROM SALES_RAW;

-- Daily revenue by date
SELECT SALE_DATE, REVENUE, QUANTITY
FROM SALES_RAW
ORDER BY SALE_DATE DESC
LIMIT 10;

-- Weekly revenue summary
SELECT
    DATE_TRUNC('WEEK', SALE_DATE) AS week_start,
    SUM(REVENUE) AS weekly_revenue,
    SUM(QUANTITY) AS weekly_quantity
FROM SALES_RAW
GROUP BY 1
ORDER BY 1;

-- Daily duplicates check
SELECT SALE_DATE, COUNT(*) AS duplicate_count
FROM SALES_RAW
GROUP BY SALE_DATE
HAVING COUNT(*) > 1;
