-- creamos la tabla con un id creciente (parecido al identity para poder hacer mods por id)
create table iris(
    id int,
    sepallengthcm float,
    sepalwidthcm float,
    petallengthcm float,
    petalwidthcm float,
    species      VARCHAR(50)
);
COPY iris
FROM '/data/iris.csv'
DELIMITER ','
CSV HEADER
NULL AS 'NA';