create table iris(
    SepalLengthCm float,
    SepalWidthCm float,
    PetalLengthCm float,
    PetalWidthCm float,
    Species      VARCHAR(50)
);
COPY iris
FROM '/data/Iris.csv'
DELIMITER ','
CSV HEADER
NULL AS 'NA';