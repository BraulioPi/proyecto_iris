-- creamos la tabla con un id creciente (parecido al identity para poder hacer mods por id)
create table iris(
    id serial Primary key,
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