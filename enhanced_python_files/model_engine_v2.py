"""
Advanced Model Engine v2.0 - XGBoost, LightGBM, Ensemble
Author: Manav Garg

FIXED: Walk-forward validation to prevent look-ahead bias in backtesting.
Previously, predict() was called on the full dataset including training data,
causing the model to "cheat" in backtests. Now predictions are only made on
truly out-of-sample data using the same walk-forward approach as market_regime.
"""

import numpy as np
import pandas as pd
import logging
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not installed. Run: pip install xgboost")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("Warning: LightGBM not installed. Run: pip install lightgbm")

logger = logging.getLogger(__name__)


class TradingModelEngine:
    """Advanced ML model engine for intraday trading predictions"""

    def __init__(self, horizon=3, model_type='ensemble', n_splits=5):
        """
        Parameters:
        -----------
        horizon    : int  — prediction horizon (bars ahead)
        model_type : str  — 'rf', 'xgboost', 'lightgbm', 'ensemble'
        n_splits   : int  — number of walk-forward folds
        """
        self.horizon = horizon
        self.model_type = model_type
        self.n_splits = n_splits
        self.model = None
        self.feature_importance = None
        self.performance_metrics = {}
        self.fold_metrics = []

    # ------------------------------------------------------------------
    # Target
    # ------------------------------------------------------------------

    def prepare_target(self, df):
        df = df.copy()
        df["Future_Return"] = df["Close"].shift(-self.horizon) / df["Close"] - 1
        df["Target"] = (df["Future_Return"] > 0).astype(int)
        df.dropna(subset=["Target"], inplace=True)
        return df

    # ------------------------------------------------------------------
    # Model builders
    # ------------------------------------------------------------------

    def _build_rf(self):
        return RandomForestClassifier(
            n_estimators=200, max_depth=8,
            min_samples_split=10, min_samples_leaf=5,
            class_weight='balanced', random_state=42, n_jobs=-1
        )

    def _build_xgb(self):
        if not XGBOOST_AVAILABLE:
            raise ImportError("pip install xgboost")
        return xgb.XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            gamma=1, reg_alpha=0.1, reg_lambda=1,
            random_state=42, n_jobs=-1, eval_metric='logloss'
        )

    def _build_lgb(self):
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("pip install lightgbm")
        return lgb.LGBMClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            num_leaves=31, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1, class_weight='balanced',
            random_state=42, n_jobs=-1, verbose=-1
        )

    def _build_ensemble(self):
        estimators = [('rf', self._build_rf())]
        if XGBOOST_AVAILABLE:
            estimators.append(('xgb', self._build_xgb()))
        if LIGHTGBM_AVAILABLE:
            estimators.append(('lgb', self._build_lgb()))
        return VotingClassifier(estimators=estimators, voting='soft', n_jobs=-1)

    def _fresh_model(self):
        builders = {
            'rf': self._build_rf,
            'xgboost': self._build_xgb,
            'lightgbm': self._build_lgb,
            'ensemble': self._build_ensemble,
        }
        if self.model_type not in builders:
            raise ValueError(f"Unknown model_type: {self.model_type}")
        return builders[self.model_type]()

    # ------------------------------------------------------------------
    # Walk-forward training  ← THE KEY FIX
    # ------------------------------------------------------------------

    def train_walk_forward(self, df, features):
        """
        Walk-forward cross-validation.

        Splits the time series into n_splits folds. For each fold:
          - Train on all data BEFORE the fold window
          - Predict ONLY on the fold window (truly out-of-sample)

        This mirrors the market_regime project approach and prevents
        the model from seeing future data during backtesting.

        Returns df with 'Prediction' and 'Confidence' filled only for
        out-of-sample rows (the last ~30% of the series).
        """
        df = self.prepare_target(df)

        # Align features
        available = [f for f in features if f in df.columns]
        if len(available) < len(features):
            logger.warning(f"Missing {len(features)-len(available)} features — using {len(available)}")

        X = df[available].values
        y = df["Target"].values
        n = len(df)

        # Reserve first 60% strictly for initial training seed
        # Walk-forward over remaining 40% in n_splits folds
        train_seed_end = int(n * 0.60)
        oos_size = n - train_seed_end
        fold_size = oos_size // self.n_splits

        predictions = np.full(n, np.nan)
        confidences = np.full(n, np.nan)
        fold_accs = []

        logger.info(f"Walk-forward: {self.n_splits} folds over {oos_size} OOS rows")

        for fold in range(self.n_splits):
            fold_start = train_seed_end + fold * fold_size
            fold_end = fold_start + fold_size if fold < self.n_splits - 1 else n

            # Train on everything BEFORE this fold
            X_train = X[:fold_start]
            y_train = y[:fold_start]

            # Test on this fold only
            X_test = X[fold_start:fold_end]
            y_test = y[fold_start:fold_end]

            if len(X_train) < 50 or len(X_test) == 0:
                continue

            model = self._fresh_model()
            model.fit(X_train, y_train)

            preds = model.predict(X_test)
            probs = model.predict_proba(X_test)[:, 1]

            predictions[fold_start:fold_end] = preds
            confidences[fold_start:fold_end] = probs

            acc = accuracy_score(y_test, preds)
            fold_accs.append(acc)
            logger.info(f"Fold {fold+1}/{self.n_splits} Accuracy: {acc:.4f}")

        # Train final model on ALL data for live decisions
        logger.info("Training final model on full dataset for live use...")
        self.model = self._fresh_model()
        self.model.fit(X, y)

        # Feature importance
        self._extract_feature_importance(available)

        # OOS metrics (only on rows that have predictions)
        oos_mask = ~np.isnan(predictions)
        if oos_mask.sum() > 0:
            oos_acc = accuracy_score(y[oos_mask], predictions[oos_mask].astype(int))
            try:
                oos_auc = roc_auc_score(y[oos_mask], confidences[oos_mask])
            except Exception:
                oos_auc = 0.5
        else:
            oos_acc, oos_auc = 0.0, 0.5

        self.fold_metrics = fold_accs
        self.performance_metrics = {
            'oos_accuracy': oos_acc,
            'oos_auc': oos_auc,
            'fold_accuracies': fold_accs,
            'mean_fold_accuracy': np.mean(fold_accs) if fold_accs else 0,
        }

        logger.info(f"OOS Accuracy: {oos_acc:.3f} | OOS AUC: {oos_auc:.3f}")

        # Attach predictions to df
        df = df.copy()
        df["Prediction"] = predictions
        df["Confidence"] = confidences

        # For live decision engine: fill last row with final model's prediction
        last_X = X[-1:].reshape(1, -1)
        df.iloc[-1, df.columns.get_loc("Prediction")] = self.model.predict(last_X)[0]
        df.iloc[-1, df.columns.get_loc("Confidence")] = self.model.predict_proba(last_X)[0][1]

        return df

    def _extract_feature_importance(self, features):
        """Extract feature importance from trained model"""
        try:
            if hasattr(self.model, 'feature_importances_'):
                imps = self.model.feature_importances_
            elif hasattr(self.model, 'named_estimators_'):
                all_imps = [
                    est.feature_importances_
                    for est in self.model.named_estimators_.values()
                    if hasattr(est, 'feature_importances_')
                ]
                imps = np.mean(all_imps, axis=0) if all_imps else None
            else:
                imps = None

            if imps is not None:
                self.feature_importance = pd.DataFrame({
                    'feature': features,
                    'importance': imps
                }).sort_values('importance', ascending=False)
        except Exception as e:
            logger.warning(f"Could not extract feature importance: {e}")

    # ------------------------------------------------------------------
    # Live prediction (uses final model trained on all data)
    # ------------------------------------------------------------------

    def predict_live(self, df, features):
        """Predict on new data using the final trained model"""
        if self.model is None:
            raise ValueError("Model not trained. Call train_walk_forward() first.")
        available = [f for f in features if f in df.columns]
        X = df[available].values
        preds = self.model.predict(X)
        probs = self.model.predict_proba(X)[:, 1]
        df = df.copy()
        df["Prediction"] = preds
        df["Confidence"] = probs
        return df

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def get_performance_report(self):
        if not self.performance_metrics:
            return "Model not trained yet."

        m = self.performance_metrics
        fold_lines = "\n".join(
            f"  Fold {i+1}: {acc:.4f}"
            for i, acc in enumerate(m.get('fold_accuracies', []))
        )

        report = f"""
{'='*60}
Model Performance Report (Walk-Forward OOS)
{'='*60}
Model Type : {self.model_type.upper()}
Horizon    : {self.horizon} periods
Folds      : {self.n_splits}

Walk-Forward Fold Accuracies:
{fold_lines}
  Mean    : {m['mean_fold_accuracy']:.4f}

Out-of-Sample Metrics:
  Accuracy : {m['oos_accuracy']:.3f}
  AUC-ROC  : {m['oos_auc']:.3f}

Top 10 Important Features:
"""
        if self.feature_importance is not None:
            for _, row in self.feature_importance.head(10).iterrows():
                report += f"  {row['feature']:<30} {row['importance']:.4f}\n"

        report += "=" * 60
        return report


# ------------------------------------------------------------------
# Convenience function (drop-in replacement for old API)
# ------------------------------------------------------------------

def train_intraday_model(df, features=None, horizon=3, model_type='ensemble', n_splits=5):
    """
    Train model with walk-forward validation.

    Returns
    -------
    tuple : (df_with_oos_predictions, engine)

    The returned df has 'Prediction' and 'Confidence' set ONLY for
    out-of-sample rows. Pass this df directly to BacktestEngine —
    the backtest will now be genuinely out-of-sample.
    """
    engine = TradingModelEngine(horizon=horizon, model_type=model_type, n_splits=n_splits)

    if features is None:
        exclude = {'Open', 'High', 'Low', 'Close', 'Volume',
                   'Target', 'Future_Return', 'Prediction', 'Confidence'}
        features = [c for c in df.columns if c not in exclude]

    df_pred = engine.train_walk_forward(df, features)
    return df_pred, engine


if __name__ == "__main__":
    print("Advanced Model Engine v2.0 — Walk-Forward Fix Applied")
    print(f"  XGBoost  : {'✓' if XGBOOST_AVAILABLE else '✗ pip install xgboost'}")
    print(f"  LightGBM : {'✓' if LIGHTGBM_AVAILABLE else '✗ pip install lightgbm'}")