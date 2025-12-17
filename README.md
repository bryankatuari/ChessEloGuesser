# Chess Elo Guesser

## Team Information

Team ID: 9
Team Members: Bryan Katuari, Owen Badgley

## Overview

This project explores the use of machine learning to predict chess player Elo ratings from basc game-level data. Features extracted from historical chess games in Lichess' public database are used to train a regression model that estimates the Elo ratings of the White and Black players. Model performance is evaluated using Mean Absolute Error (MAE) and compared against baseline predictors.

In addition to training and evaluation on a dataset of games, the project includes an implemented inference pipeline that allows the trained model to predict the Elo rating of a single chess game. Given a PGN file containing one game, the system extracts the same feature representation used during training and produces an estimated Elo rating based solely on that game’s characteristics.

## Usage

You need to have stockfish downloaded onto your computer, and put the path as STOCKFISH_PATH in a .env file. After that, install the requirements and run the files in this order:

preprocessing.py

extract_from_csv.py

train.py

make_prediction.py <- if you have a pgn file and a model to make a prediction with

## Video Code Run

https://youtu.be/xIl71BIr0EI

