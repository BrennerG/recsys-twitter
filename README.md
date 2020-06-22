# ACM RecSys Twitter Challenge 2020
[The Challenge](http://www.recsyschallenge.com/2020/)  
[Data and Leaderboard](https://recsys-twitter.com)  

* First Submission - __15.6.2020__
* Presentation - __25.6.2020__  
* Final Submission - __30.6.2020__  

# TODO
## 0 preprocess data
* [x] preprocess

| Variables  | Processing |
| ------------- | ------------- |
| Text tokens | lookup for BERT embeddings |
| Hashtags | TF-IDF transfo |
| Tweet id | do not use |
| Present media | one-hot (6 types) |
| Present links | do not use |
| Present domains | one-hot |
| Tweet type | one-hot |
| Language | one-hot |
| UNIX timestamp | scale |
| User id | do not use |
| Follower count | scale |
| Following count | scale |
| Verified | use |
| Account creation | scale |
| Engagee follows engager | use |
| Engagement timestamps (4x) | boolean |

## 1 split data
* [x] train, test, dev

## 2 create ground truth
* [x] create ground truth .csv (engaging_user_id, tweet_id, engagement) - for each interaction type
* [x] implement ["read predictions" function](https://recsys-twitter.com/code/snippets)

## 3 Implement Classifiers
### A Decision Tree Classifier
* [x] implement cheap baseline clf
* [x] evaluate (PR-AUC, Cross-Entropy-Loss)
* [x] predict whole dataset
* [x] include scores in TUWEL wiki

### B Matrix Factorization
* [ ] Prepare Features
* [ ] Train
* [ ] evaluate (PR-AUC, Cross-Entropy-Loss)

### C Content Based
* [ ] Prepare Features
* [ ] Train
* [ ] evaluate (PR-AUC, Cross-Entropy-Loss)

### D Neural network
* [ ] research network approach from the [paper](https://arxiv.org/abs/2004.13715)
* [ ] research Neural Collaborative Filtering [paper](https://arxiv.org/pdf/1708.05031.pdf)
* [ ] Prepare Features
* [ ] Train
* [ ] evaluate (PR-AUC, Cross-Entropy-Loss)

## 4 iterate
* [ ] "go beyond" collaborative filtering (ensemble?)
* [ ] tune (hyper-)parameters
* [ ] additional feature engineering
* [ ] git --gud .
* [ ] create final prediction

## 5 write report ACM style
* [ ] Intro
* [ ] Approach
* [ ] Show evaluation results
* [ ] Conclusion

## 6 submission
* [ ] report.pdf - _ACM style report_
* [ ] slides.pdf - _slides of group presentation_
* [ ] src/ - _documented code_

# Scoring
## Data Intensive Computing VU 
* [ ] Submission quality/non-trivial solution: 10p
* [ ] Correct and efficient use of Spark resources: 15p
* [ ] Code documentation: 5p
* [ ] Report: 10p
* [ ] Presentation: 10p

## Recommender Systems VU
* best effort ?
