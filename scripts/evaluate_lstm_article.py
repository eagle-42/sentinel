#!/usr/bin/env python3
"""
📊 Évaluation complète du modèle LSTM (méthode article)
Compare avec résultats article arXiv:2501.17366v1
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
import numpy as np
import torch
from loguru import logger
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer

from core.prediction import PricePredictor
from constants import CONSTANTS


def prepare_data_like_training(df: pd.DataFrame):
    """Prépare les données exactement comme l'entraînement"""
    # Période 2019-2024
    df = df[(df['DATE'] >= '2019-01-01') & (df['DATE'] <= '2024-09-30')].copy()
    
    # Features |corr| > 0.5
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    correlations = df[numeric_cols].corr()['Close'].abs()
    selected_features = correlations[correlations > 0.5].index.tolist()
    selected_features = ['DATE'] + [f for f in selected_features if f != 'Close'] + ['Close']
    df = df[selected_features].copy()
    
    # Forward-fill
    df = df.ffill()
    
    # RETURNS
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        df[f'{col}_RETURN'] = df[col].pct_change()
    
    df = df.dropna()
    df.rename(columns={'Close_RETURN': 'TARGET'}, inplace=True)
    
    return df


def evaluate_model():
    """Évaluation complète du modèle"""
    print("=" * 80)
    print("📊 ÉVALUATION COMPLÈTE LSTM - MÉTHODE ARTICLE")
    print("=" * 80)
    
    # Charger données
    df = pd.read_parquet(CONSTANTS.get_data_path("features", "SPY"))
    df_prepared = prepare_data_like_training(df)
    
    print(f"\n📊 DATASET:")
    print(f"   Période: {df_prepared['DATE'].min().date()} → {df_prepared['DATE'].max().date()}")
    print(f"   Jours: {len(df_prepared)}")
    
    # Split 60/20/20 comme entraînement
    n_train = int(len(df_prepared) * 0.6)
    n_val = int(len(df_prepared) * 0.2)
    
    df_train = df_prepared.iloc[:n_train].copy()
    df_val = df_prepared.iloc[n_train:n_train+n_val].copy()
    df_test = df_prepared.iloc[n_train+n_val:].copy()
    
    print(f"\n📊 SPLIT:")
    print(f"   Train: {len(df_train)} jours ({df_train['DATE'].min().date()} → {df_train['DATE'].max().date()})")
    print(f"   Val:   {len(df_val)} jours ({df_val['DATE'].min().date()} → {df_val['DATE'].max().date()})")
    print(f"   Test:  {len(df_test)} jours ({df_test['DATE'].min().date()} → {df_test['DATE'].max().date()})")
    
    # Extraire features RETURNS
    feature_cols = [col for col in df_prepared.columns if '_RETURN' in col or col == 'TARGET']
    if 'DATE' in feature_cols:
        feature_cols.remove('DATE')
    
    print(f"\n📊 FEATURES: {len(feature_cols)} RETURNS")
    for col in feature_cols:
        print(f"   - {col}")
    
    # Préparer données pour prédiction
    train_data = df_train[feature_cols].values
    val_data = df_val[feature_cols].values
    test_data = df_test[feature_cols].values
    
    # Scaler (fit sur train, transform sur val/test)
    imputer = SimpleImputer(strategy='mean')
    train_data = imputer.fit_transform(train_data)
    val_data = imputer.transform(val_data)
    test_data = imputer.transform(test_data)
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_data)
    val_scaled = scaler.transform(val_data)
    test_scaled = scaler.transform(test_data)
    
    # Créer séquences
    def create_sequences(data, seq_length=216):
        if len(data) < seq_length:
            return None, None
        X, y = [], []
        for i in range(seq_length, len(data)):
            X.append(data[i-seq_length:i])
            y.append(data[i, 0])  # TARGET = première colonne
        return np.array(X), np.array(y)
    
    X_val, y_val = create_sequences(val_scaled, 216)
    X_test, y_test = create_sequences(test_scaled, 216)
    
    if X_test is None:
        print("\n❌ Pas assez de données test pour window=216")
        return
    
    print(f"\n📊 SÉQUENCES:")
    print(f"   Val:  X={X_val.shape}, y={y_val.shape}")
    print(f"   Test: X={X_test.shape}, y={y_test.shape}")
    
    # Charger modèle
    predictor = PricePredictor("SPY")
    if not predictor.load_model():
        print("\n❌ Impossible de charger le modèle")
        return
    
    print(f"\n✅ Modèle chargé: LSTM[64x2] + Dense[32→1]")
    
    # Prédictions sur VAL
    print(f"\n" + "=" * 80)
    print("📊 ÉVALUATION VALIDATION SET")
    print("=" * 80)
    
    val_preds_scaled = []
    with torch.no_grad():
        for i in range(len(X_val)):
            seq = torch.FloatTensor(X_val[i:i+1]).to(predictor.device)
            pred = predictor.model(seq).cpu().numpy()[0, 0]
            val_preds_scaled.append(pred)
    
    val_preds_scaled = np.array(val_preds_scaled)
    
    # Métriques sur RETURNS
    mae_returns_val = np.mean(np.abs(val_preds_scaled - y_val))
    rmse_returns_val = np.sqrt(np.mean((val_preds_scaled - y_val) ** 2))
    mape_returns_val = np.mean(np.abs((val_preds_scaled - y_val) / (y_val + 1e-8))) * 100
    
    print(f"\n📊 MÉTRIQUES SUR RETURNS (Val):")
    print(f"   MAE (returns):  {mae_returns_val:.6f}")
    print(f"   RMSE (returns): {rmse_returns_val:.6f}")
    print(f"   MAPE:           {mape_returns_val:.2f}%")
    
    # Convertir en PRIX pour comparer avec article
    # Dénormaliser returns
    val_dummy = np.zeros((len(val_preds_scaled), len(feature_cols)))
    val_dummy[:, 0] = val_preds_scaled
    val_returns_denorm = scaler.inverse_transform(val_dummy)[:, 0]
    
    y_val_dummy = np.zeros((len(y_val), len(feature_cols)))
    y_val_dummy[:, 0] = y_val
    y_val_returns_denorm = scaler.inverse_transform(y_val_dummy)[:, 0]
    
    # Reconstruire prix depuis returns
    # Prix[t] = Prix[t-1] * (1 + return[t])
    val_start_idx = n_train + 216
    val_prices_real = df_prepared['Close'].iloc[val_start_idx:val_start_idx+len(y_val)].values
    val_prices_pred = []
    
    for i, ret in enumerate(val_returns_denorm):
        if i == 0:
            # Premier prix = prix réel précédent * (1 + return prédit)
            prev_price = df_prepared['Close'].iloc[val_start_idx + i - 1]
            pred_price = prev_price * (1 + ret)
        else:
            # Utiliser prix réel précédent pour éviter accumulation erreurs
            prev_price = val_prices_real[i-1]
            pred_price = prev_price * (1 + ret)
        val_prices_pred.append(pred_price)
    
    val_prices_pred = np.array(val_prices_pred)
    
    # Métriques sur PRIX
    mae_price_val = np.mean(np.abs(val_prices_pred - val_prices_real))
    rmse_price_val = np.sqrt(np.mean((val_prices_pred - val_prices_real) ** 2))
    mape_price_val = np.mean(np.abs((val_prices_pred - val_prices_real) / val_prices_real)) * 100
    accuracy_val = 100 - mape_price_val
    
    print(f"\n📊 MÉTRIQUES SUR PRIX (Val):")
    print(f"   MAE (USD):  {mae_price_val:.2f}$")
    print(f"   RMSE (USD): {rmse_price_val:.2f}$")
    print(f"   MAPE:       {mape_price_val:.2f}%")
    print(f"   Accuracy:   {accuracy_val:.2f}%")
    
    # Prédictions sur TEST
    print(f"\n" + "=" * 80)
    print("📊 ÉVALUATION TEST SET")
    print("=" * 80)
    
    test_preds_scaled = []
    with torch.no_grad():
        for i in range(len(X_test)):
            seq = torch.FloatTensor(X_test[i:i+1]).to(predictor.device)
            pred = predictor.model(seq).cpu().numpy()[0, 0]
            test_preds_scaled.append(pred)
    
    test_preds_scaled = np.array(test_preds_scaled)
    
    # Métriques sur RETURNS
    mae_returns_test = np.mean(np.abs(test_preds_scaled - y_test))
    rmse_returns_test = np.sqrt(np.mean((test_preds_scaled - y_test) ** 2))
    
    print(f"\n📊 MÉTRIQUES SUR RETURNS (Test):")
    print(f"   MAE (returns):  {mae_returns_test:.6f}")
    print(f"   RMSE (returns): {rmse_returns_test:.6f}")
    
    # Convertir en PRIX
    test_dummy = np.zeros((len(test_preds_scaled), len(feature_cols)))
    test_dummy[:, 0] = test_preds_scaled
    test_returns_denorm = scaler.inverse_transform(test_dummy)[:, 0]
    
    test_start_idx = n_train + n_val + 216
    test_prices_real = df_prepared['Close'].iloc[test_start_idx:test_start_idx+len(y_test)].values
    test_prices_pred = []
    
    for i, ret in enumerate(test_returns_denorm):
        if i == 0:
            prev_price = df_prepared['Close'].iloc[test_start_idx + i - 1]
            pred_price = prev_price * (1 + ret)
        else:
            prev_price = test_prices_real[i-1]
            pred_price = prev_price * (1 + ret)
        test_prices_pred.append(pred_price)
    
    test_prices_pred = np.array(test_prices_pred)
    
    # Métriques sur PRIX
    mae_price_test = np.mean(np.abs(test_prices_pred - test_prices_real))
    rmse_price_test = np.sqrt(np.mean((test_prices_pred - test_prices_real) ** 2))
    mape_price_test = np.mean(np.abs((test_prices_pred - test_prices_real) / test_prices_real)) * 100
    accuracy_test = 100 - mape_price_test
    
    print(f"\n📊 MÉTRIQUES SUR PRIX (Test):")
    print(f"   MAE (USD):  {mae_price_test:.2f}$")
    print(f"   RMSE (USD): {rmse_price_test:.2f}$")
    print(f"   MAPE:       {mape_price_test:.2f}%")
    print(f"   Accuracy:   {accuracy_test:.2f}%")
    
    # Dernière prédiction
    last_real = test_prices_real[-1]
    last_pred = test_prices_pred[-1]
    last_error = abs(last_real - last_pred)
    
    print(f"\n💰 DERNIÈRE PRÉDICTION (Test):")
    print(f"   Prix réel:  {last_real:.2f}$")
    print(f"   Prix prédit: {last_pred:.2f}$")
    print(f"   Écart:      {last_error:.2f}$ ({last_error/last_real*100:.2f}%)")
    
    # COMPARAISON ARTICLE
    print(f"\n" + "=" * 80)
    print("📊 COMPARAISON AVEC ARTICLE (arXiv:2501.17366v1)")
    print("=" * 80)
    
    print(f"\n{'Métrique':<15} | {'Article (SPX)':<20} | {'Notre Modèle (SPY)':<20}")
    print("-" * 60)
    print(f"{'MAE':<15} | {'175.9':<20} | {f'{mae_price_test:.2f}':<20}")
    print(f"{'RMSE':<15} | {'207.34':<20} | {f'{rmse_price_test:.2f}':<20}")
    print(f"{'Accuracy':<15} | {'96.41%':<20} | {f'{accuracy_test:.2f}%':<20}")
    print(f"{'Période':<15} | {'2013-2024 (11 ans)':<20} | {'2019-2024 (5 ans)':<20}")
    print(f"{'Window':<15} | {'216':<20} | {'216':<20}")
    print(f"{'Features':<15} | {'CLOSE only':<20} | {f'{len(feature_cols)} RETURNS':<20}")
    
    # Verdict
    print(f"\n" + "=" * 80)
    print("🎯 VERDICT")
    print("=" * 80)
    
    if accuracy_test > 95:
        print("🏆🏆🏆 EXCELLENT ! Accuracy > 95% (niveau article)")
    elif accuracy_test > 90:
        print("🏆🏆 TRÈS BON ! Accuracy > 90%")
    elif accuracy_test > 85:
        print("🏆 BON ! Accuracy > 85%")
    else:
        print("⚠️ Acceptable mais peut être amélioré")
    
    if mae_price_test < 200:
        print(f"✅ MAE < 200$ = Prédictions précises")
    
    if last_error < 50:
        print(f"✅ Dernière prédiction < 50$ d'erreur = Excellent")
    
    print("\n" + "=" * 80)
    
    return {
        'val_mae_price': mae_price_val,
        'val_accuracy': accuracy_val,
        'test_mae_price': mae_price_test,
        'test_accuracy': accuracy_test
    }


if __name__ == "__main__":
    evaluate_model()
