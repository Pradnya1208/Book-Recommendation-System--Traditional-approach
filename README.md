<div align="center">
  
[1]: https://github.com/Pradnya1208
[2]: https://www.linkedin.com/in/pradnya-patil-b049161ba/
[3]: https://public.tableau.com/app/profile/pradnya.patil3254#!/
[4]: https://twitter.com/Pradnya1208


[![github](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/c292abd3f9cc647a7edc0061193f1523e9c05e1f/icons/git.svg)][1]
[![linkedin](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/9f5c4a255972275ced549ea6e34ef35019166944/icons/iconmonstr-linkedin-5.svg)][2]
[![tableau](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/e257c5d6cf02f13072429935b0828525c601414f/icons/icons8-tableau-software%20(1).svg)][3]
[![twitter](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/c9f9c5dc4e24eff0143b3056708d24650cbccdde/icons/iconmonstr-twitter-5.svg)][4]

</div>

# <div align="center">Book Recommendation System</div>
<div align = "center"> <img src = "https://github.com/Pradnya1208/Book-Recommendation-System/blob/main/output/gif.gif?raw=true"></div>


## Overview:
What is recommender System?

- Based on previous(past) behaviours, it predicts the likelihood that a user would prefer an item.
- For example, Netflix uses recommendation system. It suggest people new movies according to their past activities that are like watching and voting movies.
- The purpose of recommender systems is recommending new things that are not seen before from people.

### Recommender systems:
#### Content based filtering:
- Content-Based recommender system tries to guess the features or behavior of a user given the item’s features, he/she reacts positively to.
- Once, we know the likings of the user we can embed user's choice in an embedding space using the feature vector generated and recommend accordingly.
#### collaborative filtering:
- Collaborative does not need the features of the items to be given. Every user and item is described by a feature vector or embedding.
- It creates embedding for both users and items on its own. It embeds both users and items in the same embedding space.
- It considers other users’ reactions while recommending a particular user. It notes which items a particular user likes and also the items that the users with behavior and likings like him/her likes, to recommend items to that user.
- It collects user feedbacks on different items and uses them for recommendations.

#### Model-based collaborative filtering
- Remembering the matrix is not required here. From the matrix, we try to learn how a specific user or an item behaves. We compress the large interaction matrix using dimensional Reduction or using clustering algorithms. In this type, We fit machine learning models and try to predict how many ratings will a user give a product. 
- There are several methods:<br>
`Clustering algorithms`<br>
`Matrix Factorization based algorithm`<br>
`Deep Learning methods`<br>

#### Similarity Metrics:
They are mathematical measures which are used to determine how similar is a vector to a given vector.
Similarity metrics used mostly:<br>
- **Cosine Similarity**: The Cosine angle between the vectors.
- **Dot Product**: The cosine angle and magnitude of the vectors also matters.
- **Euclidian Distance**: The elementwise squared distance between two vectors
- **Pearson Similarity**: It is a coefficient given by:
<img src = "https://github.com/Pradnya1208/Book-Recommendation-System/blob/main/output/pearson.PNG?raw=true" width="20%">



## Objective:
This project aims at finding the multiple ways that our data can recommend books.
## Dataset:
[Book-Crossing: User review ratings](https://www.kaggle.com/ruchi798/bookcrossing-dataset)

This dataset Contains 278,858 users (anonymized but with demographic information) providing 1,149,780 ratings (explicit / implicit) about 271,379 books.

### Data Fields:
| Field             | Description                                                                |
| ----------------- | ------------------------------------------------------------------ |
| User-ID | Book reader's unique user ID|
| ISBN | ISBN of book|
| Book Rating | Book rating by individual user|
| Book Title | Book title|
| Book Author | Book author|
| Publisher | Book publisher|
| Age | Age of user|
| City | City where user is from|
| State | State where user is from|
| Country | Country where user is from|
| Category | Book Category|
| Language | Language of the book|
| Summary | Short summary about the book|
| img_* | Book cover image|



[1]: https://github.com/Pradnya1208
[2]: https://www.linkedin.com/in/pradnya-patil-b049161ba/
[3]: https://public.tableau.com/app/profile/pradnya.patil3254#!/
[4]: https://twitter.com/Pradnya1208


[![github](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/c292abd3f9cc647a7edc0061193f1523e9c05e1f/icons/git.svg)][1]
[![linkedin](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/9f5c4a255972275ced549ea6e34ef35019166944/icons/iconmonstr-linkedin-5.svg)][2]
[![tableau](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/e257c5d6cf02f13072429935b0828525c601414f/icons/icons8-tableau-software%20(1).svg)][3]
[![twitter](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/c9f9c5dc4e24eff0143b3056708d24650cbccdde/icons/iconmonstr-twitter-5.svg)][4]
