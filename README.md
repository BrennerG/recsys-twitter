# ACM RecSys Twitter Challenge 2020
[The Challenge](http://www.recsyschallenge.com/2020/)  
[Data and Leaderboard](https://recsys-twitter.com)  

* First Submission - __15.6.2020__
* Presentation - __25.6.2020__  
* Final Submission - __30.6.2020__  

# TODO
## 0 preprocess data
* [ ] preprocess

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
* [ ] train, test, dev

## 2 create ground truth
* [ ] create ground truth .csv (engaging_user_id, tweet_id, engagement) - for each interaction type
* [ ] implement ["read predictions" function](https://recsys-twitter.com/code/snippets)

## 3 create ratings matrix
(for collaborative filtering approaches)  
one matrix for each engagement type
* [ ] like matrix
* [ ] reply matrix
* [ ] retweet matrix
* [ ] retweet with comment matrix

## 4 create adjacency matrix
* [ ] parse follower-following graph (engaged-follows-engaging)
* [ ] how can you use this information?

## 5 implement baseline classifier
* [ ] implement cheap baseline clf
* [ ] evaluate (PR-AUC, Cross-Entropy-Loss)
* [ ] predict whole dataset
* [ ] include scores in TUWEL wiki

## 6 neural network
* [ ] implement neural network approach from the [paper](https://arxiv.org/abs/2004.13715)
* [ ] evaluate (PR-AUC, Cross-Entropy-Loss)

## 7 iterate
* [ ] "go beyond" collaborative filtering (ensemble?)
* [ ] tune (hyper-)parameters
* [ ] additional feature engineering
* [ ] git --gud .
* [ ] create final prediction

## 8 write report ACM style
* [ ] Intro
* [ ] Approach
* [ ] Show evaluation results
* [ ] Conclusion

## 9 submission
* [ ] report.pdf - _ACM style report_
* [ ] slides.pdf - _slides of group presentation_
* [ ] src/ - _documented code_

# Scoring
## Data Intensive Computing VU 
* Submission quality/non-trivial solution: 10p
* Correct and efficient use of Spark resources: 15p
* Code documentation: 5p
* Report: 10p
* Presentation: 10p

## Recommender Systems VU
* best effort ?
