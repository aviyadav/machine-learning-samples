"""
Mastering Probability Distributions
====================================
A hands-on walkthrough using the ShopSmart customer dataset.

Steps:
  1. Generate synthetic dataset (9 distributions)
  2. Visualise every distribution
  3. Fit distributions & run goodness-of-fit tests
  4. Build a churn-prediction model
  5. Bayesian updating with the Beta distribution
  6. Poisson regression for count data
"""

# ══════════════════════════════════════════════════════════════
# Imports
# ══════════════════════════════════════════════════════════════
import numpy as np
import polars as pl
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import beta as beta_dist
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import statsmodels.api as sm
import statsmodels.formula.api as smf
import warnings
from execution_profiler import profile_execution

warnings.filterwarnings('ignore')


# ══════════════════════════════════════════════════════════════
# Step 1: Generating the Dataset
# ══════════════════════════════════════════════════════════════


def step1_generate_dataset(n=10000, seed=42):
    """Generate the ShopSmart synthetic customer dataset using 9 distributions."""
    np.random.seed(seed)

    # 1. NORMAL — Customer Age
    age = np.random.normal(loc=35, scale=10, size=n).clip(18, 75).astype(int)

    # 2. BERNOULLI — Churned (1 = left, 0 = stayed)
    churned = np.random.binomial(n=1, p=0.25, size=n)

    # 3. BINOMIAL — Purchases in Last Month (max 20)
    purchases_last_month = np.random.binomial(n=20, p=0.15, size=n)

    # 4. GEOMETRIC — Days Until First Purchase
    days_until_first_purchase = np.random.geometric(p=0.2, size=n)

    # 5. POISSON — Support Tickets Filed
    support_tickets = np.random.poisson(lam=2.5, size=n)

    # 6. EXPONENTIAL — Days Between Purchases
    time_between_purchases = np.random.exponential(scale=7, size=n)  # avg 7 days

    # 7. GAMMA — Session Duration (minutes)
    session_duration = np.random.gamma(shape=3, scale=4, size=n)  # mean = 12 mins

    # 8. BETA — Click-Through Rate (0 to 1)
    click_through_rate = np.random.beta(a=2, b=8, size=n)  # skewed low ~0.2 mean

    # 9. UNIFORM — Discount Applied (5% to 50%)
    discount_used = np.random.uniform(low=5, high=50, size=n)

    # Assemble Polars DataFrame
    df = pl.DataFrame({
        'age': age,
        'churned': churned,
        'purchases_last_month': purchases_last_month,
        'days_until_first_purchase': days_until_first_purchase,
        'support_tickets': support_tickets,
        'time_between_purchases': np.round(time_between_purchases, 2),
        'session_duration': np.round(session_duration, 2),
        'click_through_rate': np.round(click_through_rate, 4),
        'discount_used': np.round(discount_used, 2)
    })

    print(df.head())
    print("\nShape:", df.shape)
    print("\nBasic Stats:\n", df.describe())

    return df


# ══════════════════════════════════════════════════════════════
# Step 2: Visualising Every Distribution
# ══════════════════════════════════════════════════════════════

def step2_visualise_distributions(df):
    """Plot histograms for all 9 features in a 3×3 grid."""
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle("ShopSmart Customer Dataset — Distribution Overview",
                 fontsize=16, fontweight='bold')

    dist_info = [
        ('age',                       'Normal — Customer Age',               'steelblue'),
        ('churned',                   'Bernoulli — Churned',                 'coral'),
        ('purchases_last_month',      'Binomial — Monthly Purchases',        'mediumseagreen'),
        ('days_until_first_purchase', 'Geometric — Days to First Purchase',  'orchid'),
        ('support_tickets',           'Poisson — Support Tickets',           'tomato'),
        ('time_between_purchases',    'Exponential — Days Between Orders',   'goldenrod'),
        ('session_duration',          'Gamma — Session Duration (min)',       'teal'),
        ('click_through_rate',        'Beta — Click-Through Rate',           'slateblue'),
        ('discount_used',             'Uniform — Discount Applied (%)',      'darkorange'),
    ]

    for ax, (col, title, color) in zip(axes.flatten(), dist_info):
        ax.hist(df[col].to_numpy(), bins=40, color=color, alpha=0.8, edgecolor='white')
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.set_xlabel(col)
        ax.set_ylabel('Frequency')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig('shopsmart_distributions.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Plot saved!")


# ══════════════════════════════════════════════════════════════
# Step 3: Fitting Distributions to Confirm Our Data
# ══════════════════════════════════════════════════════════════

def step3_fit_distributions(df):
    """Fit Normal and Gamma distributions, then run KS goodness-of-fit tests."""
    # Test: Does 'age' truly follow Normal?
    age_data = df['age'].to_numpy().astype(float)
    mu, sigma = stats.norm.fit(age_data)
    print(f"Age — Fitted Normal: μ={mu:.2f}, σ={sigma:.2f}")

    ks_stat, p_value = stats.kstest(age_data, stats.norm(loc=mu, scale=sigma).cdf)
    print(f"KS Test p-value for age: {p_value:.4f}")
    print("→ Normal fit accepted!" if p_value > 0.05 else "→ Normal fit rejected.")
    print()

    # Test: Does 'session_duration' follow Gamma?
    session_data = df['session_duration'].to_numpy()
    shape, loc, scale = stats.gamma.fit(session_data, floc=0)
    print(f"Session Duration — Fitted Gamma: shape={shape:.2f}, scale={scale:.2f}")

    ks_stat, p_value = stats.kstest(session_data,
                                     stats.gamma(a=shape, loc=loc, scale=scale).cdf)
    print(f"KS Test p-value for session_duration: {p_value:.4f}")
    print("→ Gamma fit accepted!" if p_value > 0.05 else "→ Gamma fit rejected.")


# ══════════════════════════════════════════════════════════════
# Step 4: Using Distributions to Build a Churn Prediction Model
# ══════════════════════════════════════════════════════════════

def step4_churn_prediction(df):
    """Train Logistic Regression and Random Forest models for churn prediction."""
    # ── Feature Engineering using Distribution Knowledge ──
    # Log-transform skewed features (Exponential, Gamma, Geometric)
    # to make them more Normal — this helps linear models
    df_feat = df.with_columns(
        pl.col('time_between_purchases').log1p().alias('log_time_between'),
        pl.col('session_duration').log1p().alias('log_session'),
        pl.col('days_until_first_purchase').cast(pl.Float64).log1p().alias('log_days_first'),
    ).drop('time_between_purchases', 'session_duration', 'days_until_first_purchase')

    # Features and target
    y = df_feat['churned'].to_numpy()
    X_df = df_feat.drop('churned')
    X = X_df.to_numpy().astype(float)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # ── Model 1: Logistic Regression (assumes Normal features) ──
    lr = LogisticRegression(random_state=42, max_iter=500)
    lr.fit(X_train_scaled, y_train)
    y_pred_lr = lr.predict(X_test_scaled)
    y_prob_lr = lr.predict_proba(X_test_scaled)[:, 1]

    print("=== Logistic Regression ===")
    print(classification_report(y_test, y_pred_lr))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_prob_lr):.4f}")

    # ── Model 2: Random Forest (distribution-agnostic) ──
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    y_prob_rf = rf.predict_proba(X_test)[:, 1]

    print("\n=== Random Forest ===")
    print(classification_report(y_test, y_pred_rf))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_prob_rf):.4f}")


# ══════════════════════════════════════════════════════════════
# Step 5: Bayesian Thinking with the Beta Distribution
#         — Updating Click-Through Rate Beliefs
# ══════════════════════════════════════════════════════════════

def step5_bayesian_beta_update():
    """Demonstrate Bayesian updating of CTR belief using the Beta distribution."""
    # ── Prior: Before seeing any data, we believe CTR ~ Beta(2, 8)
    # This encodes: "We think CTR is around 20%, but we're not sure"
    alpha_prior, beta_prior = 2, 8

    # Observed data: 35 clicks out of 200 impressions
    clicks = 35
    impressions = 200
    non_clicks = impressions - clicks

    # ── Posterior: After seeing data, update the Beta distribution
    # Beta posterior = Beta(alpha + clicks, beta + non_clicks)
    alpha_post = alpha_prior + clicks
    beta_post  = beta_prior + non_clicks

    x = np.linspace(0, 0.6, 300)
    prior     = beta_dist.pdf(x, alpha_prior, beta_prior)
    posterior = beta_dist.pdf(x, alpha_post, beta_post)

    plt.figure(figsize=(10, 5))
    plt.plot(x, prior, 'b--', linewidth=2,
             label=f'Prior: Beta({alpha_prior},{beta_prior})')
    plt.plot(x, posterior, 'r-', linewidth=2.5,
             label=f'Posterior: Beta({alpha_post},{beta_post})')
    plt.axvline(clicks/impressions, color='green', linestyle=':', linewidth=1.5,
                label=f'Observed CTR: {clicks/impressions:.2%}')
    plt.fill_between(x, posterior, alpha=0.15, color='red')
    plt.title('Bayesian Update of Click-Through Rate Belief\n(Beta Distribution)',
              fontsize=13, fontweight='bold')
    plt.xlabel('Click-Through Rate')
    plt.ylabel('Probability Density')
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig('bayesian_beta_update.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Credible interval
    ci_low, ci_high = beta_dist.ppf([0.025, 0.975], alpha_post, beta_post)
    print(f"Posterior Mean CTR: {alpha_post/(alpha_post+beta_post):.2%}")
    print(f"95% Credible Interval: [{ci_low:.2%}, {ci_high:.2%}]")


# ══════════════════════════════════════════════════════════════
# Step 6: Poisson Regression — When Your Target Is a Count
# ══════════════════════════════════════════════════════════════

def step6_poisson_regression(df):
    """Compare Poisson GLM vs OLS for predicting support-ticket counts."""
    # statsmodels requires pandas, so convert at the boundary
    df_pd = df.to_pandas()

    # Predicting number of support tickets (Poisson target)
    # Standard linear regression would be WRONG here —
    # counts can't be negative and aren't Normally distributed
    poisson_model = smf.glm(
        formula='support_tickets ~ age + purchases_last_month + click_through_rate',
        data=df_pd,
        family=sm.families.Poisson()
    ).fit()
    print(poisson_model.summary())

    # Compare: what would happen with OLS (ignoring distribution)?
    ols_model = smf.ols(
        formula='support_tickets ~ age + purchases_last_month + click_through_rate',
        data=df_pd
    ).fit()

    # Predict for a new customer
    new_customer = pd.DataFrame({
        'age': [28],
        'purchases_last_month': [5],
        'click_through_rate': [0.25]
    })

    poisson_pred = poisson_model.predict(new_customer)
    ols_pred = ols_model.predict(new_customer)

    print(f"\nPoisson Regression Prediction: {poisson_pred.values[0]:.2f} tickets")
    print(f"OLS (Linear) Prediction: {ols_pred.values[0]:.2f} tickets")
    print("\n→ Both may be similar, but Poisson regression is theoretically")
    print("  correct and never predicts negative counts.")


# ══════════════════════════════════════════════════════════════
# Main — Execute All Steps
# ══════════════════════════════════════════════════════════════

@profile_execution
def main():
    """Run the full probability-distributions walkthrough."""
    df = step1_generate_dataset()
    step2_visualise_distributions(df)
    step3_fit_distributions(df)
    step4_churn_prediction(df)
    step5_bayesian_beta_update()
    step6_poisson_regression(df)


if __name__ == '__main__':
    main()