-- Create users table if it doesn't already exist
CREATE TABLE IF NOT EXISTS users (
    user_id INT PRIMARY KEY,
    user_name VARCHAR(255)  -- Optional, based on whether you have user names
);

-- Create recommendations table with an array of recommended movie IDs
CREATE TABLE IF NOT EXISTS recommendations (
    user_id INT,  -- Foreign key referencing users table
    movie_id INT,  -- Movie ID
    imdb_id VARCHAR(20),  -- IMDb ID
    tmdb_id INT,  -- TMDb ID
    title VARCHAR(255),  -- Movie Title
    rating FLOAT,  -- Rating
    PRIMARY KEY (user_id, movie_id)  -- Ensure unique user-movie pairs
);


INSERT INTO recommendations (user_id, movie_id, imdb_id, tmdb_id, title, rating)
VALUES
    (1, 3379, '0053137', 35412, 'On the Beach (1959)', 5.9749503),
    (1, 132333, '3149640', 278990, 'Seve (2014)', 5.9546223),
    (1, 5490, '0074205', 19133, 'The Big Bus (1976)', 5.9546223),
    (1, 33649, '0384504', 19316, 'Saving Face (2004)', 5.7459326),
    (1, 6201, '0091374', 21867, 'Lady Jane (1986)', 5.7175446),
    (2, 3379, '0053137', 35412, 'On the Beach (1959)', 5.0166187),
    (2, 33649, '0384504', 19316, 'Saving Face (2004)', 4.8618107),
    (2, 7121, '0041090', 25431, 'Adams Rib (1949)', 4.8423276),
    (2, 132333, '3149640', 278990, 'Seve (2014)', 4.790345),
    (2, 5490, '0074205', 19133, 'The Big Bus (1976)', 4.790345)
RETURNING *;



-- select * from public.recommendations;

-- SELECT DISTINCT r.user_id, r.movie_id, r.title, r.rating
--                 FROM (
--                     SELECT r.user_id, r.movie_id, r.title, r.rating
--                     FROM recommendations r
--                     ORDER BY r.user_id, RANDOM()
--                     LIMIT 5
--                 ) AS r;