# Predicting Movie Choices Using Sequential Deep Learning
**STAT 362 — Fall 2025**

**Author:** Kevin Li

## Problem and Project Overview
The objective of this project is to predict the next movie a person will watch, given their chronologically ordered viewing history. This creates a multi-class classification problem over the possible movie classes, where the goal is to identify the correct next movie among all candidates. The model treats next-movie prediction as a sequential modeling task, leveraging an LSTM to capture temporal patterns in user behavior. This project was motivated by the fact that recommendation algorithms power many of today’s most widely used digital platforms, such as Netflix, YouTube, and Spotify.

## Dataset
This project uses the [MovieLens (latest-small)](https://files.grouplens.org/datasets/movielens/ml-latest-small-README.html) dataset, which contains 100,836 timestamped ratings from 610 users across 9,724 movies. Ratings range from 0.5 to 5.0 in increments of 0.5, and each movie includes additional metadata such as 19 possible genres and 1,589 different tags. For sequence modeling, each user’s history is sorted chronologically, then split into train/validation/test sets using a leave-one-out strategy.

**Note:** To acquire the data used in this project, please visit this [link](https://grouplens.org/datasets/movielens/) and download the file titled `ml-latest-small.zip`.

## Model Overview
### Baseline Models `(notebooks/baseline_model.ipynb)`
1. ItemPop (Popularity Baseline)
   - This model ranks movies solely by their overall frequency in the training set, providing a simple, non-personalized baseline that reflects popularity trends.
2. Milestone LSTM Model
   - This initial model takes only the sequence of movie IDs as input, without incorporating genre or rating context. While it captures some sequential structure, it suffers from severe overfitting and therefore underperformed during evaluation (see `Key Results` section for a visualization of training vs. validation accuracy).
  
### Final Model `(notebooks/final_model.ipynb)`
- An LSTM was used because next-movie prediction is inherently sequential, and LSTMs are well-suited for capturing sequential structure without succumbing to the vanishing gradient problem.
- The model accepts three inputs: the movie ID sequence, a 19-dimensional genre vector, and a normalized rating between 0 and 1 for each time step.
- The architecture consists of a 256-unit LSTM layer followed by dense layers of sizes 512, 256, and 128.
- Regularization includes a dropout rate of 0.1 and early stopping based on validation HR@10. The model is trained with the Adam optimizer using a batch size of 256.

## Key Results: Metrics and Visualizations
| Model | HR@5 (Negative Sampling) | HR@10 (Negative Sampling) |
|---|---|---|
| ItemPop | 40.66% | 57.87% |
| Milestone Model | 47.21% | 62.62% |
| Final Model | 56.07% | 69.51% |

### Insights
- The final context-aware LSTM model outperforms both baselines at every value of K (K = 1-10), demonstrating that additional context can be helpful for next-movie prediction.
- The milestone LSTM model provides a small but consistent improvement over the ItemPop baseline, demonstrating that even a simple sequence model can capture temporal patterns.

### Hit Ratio @ K
<img src="figures/Hit%20Ratio%20@%20K.png" width=45%>

### Milestone Model: Training and Validation Curves
<img src="figures/Milestone%20Model%20-%20Training%20and%20Validation%20Curves.png" width=45%>

### Final Model: Training and Validation Curves
<img src="figures/Final%20Model%20-%20Training%20and%20Validation%20Curves.png" width=45%>

## How to Run the Code
This repo includes two types of notebooks:
1. `project_demo.ipynb`: a demo that trains the final LSTM architecture and plots HR@K (K = 1–10).
2. `notebooks/`: full project notebooks (`baseline_model.ipynb` and `final_model.ipynb`) that reproduce the full experiments and visualizations from this repo.

**Note:** These instructions assume you are in the root directory of the repo and have downloaded the data from MovieLens.

**Note:** It is recommended to run these notebooks in Google Colab in order to use their free NVIDIA Tesla T4 GPU.
1. **Pre-requisite: create a virtual environment**
   ```
   python3 -m venv .venv
   ```
2. **Activate the virtual environment**
   ```
   source .venv/bin/activate
   ```
4. **Install dependencies**
   ```
   pip install -r requirements.txt
   ```
   If you plan to run the notebooks using Jupyter, install it as well:
   ```
   pip install jupyter
   ```
6. **Run the notebooks**
   - Option A: Run the notebooks in VS Code
        - Open any `.ipynb` file (e.g. `project_demo.ipynb`)
        - When VS Code prompts you to select a kernel, choose the `.venv` environment
        - Click "Run All" to execute the notebook
   - Option B: Run the notebooks in Jupyter Notebook
        - Launch Jupyter Notebook
          ```
          jupyter notebook
          ```
        - In the browser window, open `project_demo.ipynb` to run the demo or open `notebooks/baseline_model.ipynb` / `notebooks/final_model.ipynb` for full project notebooks
        - Click "Run" --> "Run All Cells"

## References
- [MovieLens Dataset](https://files.grouplens.org/datasets/movielens/ml-latest-small-README.html)
    - Harper, F. Maxwell, and Joseph A. Konstan. “The MovieLens Datasets.” ACM Transactions on Interactive Intelligent Systems, vol. 5, no. 4, 22 Dec. 2015, pp. 1–19, https://doi.org/10.1145/2827872.
- [Neural Collaborative Filtering](https://arxiv.org/abs/1708.05031)
    - He, Xiangnan, et al. “Neural Collaborative Filtering.” ArXiv:1708.05031 [Cs], 16 Aug. 2017, arxiv.org/abs/1708.05031.
