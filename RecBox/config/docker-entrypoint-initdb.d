-- init.sql: SQL script to create tables

CREATE TABLE IF NOT EXISTS users (
    user_id INT PRIMARY KEY,
    user_name VARCHAR(255)
);

CREATE TABLE IF NOT EXISTS recommendations (
    user_id INT REFERENCES users(user_id),
    movie_title VARCHAR(255),
    image_url VARCHAR(255)
);
