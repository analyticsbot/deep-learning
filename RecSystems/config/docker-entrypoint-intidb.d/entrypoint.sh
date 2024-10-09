CREATE TABLE movies (
    id SERIAL PRIMARY KEY,
    title VARCHAR(255) NOT NULL,
    year INTEGER,
    genre VARCHAR(100)
);

INSERT INTO movies (title, year, genre) VALUES
('Inception', 2010, 'Sci-Fi'),
('The Dark Knight', 2008, 'Action'),
('Interstellar', 2014, 'Adventure'),
('Avatar', 2009, 'Action'),
('Titanic', 1997, 'Romance');
