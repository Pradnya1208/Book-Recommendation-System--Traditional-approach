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


## Overview:
What is recommender System?

- Based on previous(past) behaviours, it predicts the likelihood that a user would prefer an item.
- For example, Netflix uses recommendation system. It suggest people new movies according to their past activities that are like watching and voting movies.
- The purpose of recommender systems is recommending new things that are not seen before from people.

### Recommender systems:
#### User based collaborative filtering:
- Collaborative filtering is making recommendations according to combination of your experience and experiences of other people.
- In this system, each row of matrix is user. Therefore, comparing and finding similarity between of them is computationaly hard and spend too much computational power.
- Also, habits of people can be changed. Therefore making correct and useful recommendation can be hard in time.
- In order to solve the shortcomings of this system we use Item based collaborative filtering.
#### Item based collaborative filtering:
- In this system, instead of finding relationship between users, used items like movies or stuffs are compared with each others.
- In user based recommendation systems, habits of users can be changed. This situation makes hard to recommendation. However, in item based recommendation systems, movies or stuffs does not change. Therefore recommendation is easier.


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

