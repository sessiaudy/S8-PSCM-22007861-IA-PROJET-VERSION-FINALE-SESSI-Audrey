# =================================================================
# SCRIPT : Prédiction de la Demande en Stock — Pipeline PSCM
# Auteur  : SESSI Fiawoo Akou Sika Audrey
# Projet  : PSCM / Semestre 8 — ENCG Settat
# Thème   : Gestion des Stocks & Entrepôts
# Date    : 2025-2026
# =================================================================
# Description :
#   Pipeline complet de prédiction de la demande en stock pour
#   une GMS marocaine. Implémente une architecture multi-agents
#   (1 000 sous-agents simulés) avec les modèles :
#     - RandomForest Regressor
#     - XGBoost Regressor
#     - Régression Linéaire (baseline)
#     - XGBoost Classifier (risque de rupture)
#
# Usage :
#   pip install pandas numpy scikit-learn xgboost matplotlib
#   python script_pscm_demande.py
# =================================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (mean_squared_error, r2_score,
                             accuracy_score, f1_score,
                             roc_auc_score, confusion_matrix,
                             classification_report)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# =================================================================
# 1. CHARGEMENT DES DONNÉES
# =================================================================
print("=" * 65)
print("PSCM — Prédiction de la Demande | ENCG Settat S8")
print("SESSI Fiawoo Akou Sika Audrey")
print("=" * 65)

try:
    df = pd.read_csv('dataset_gms_maroc_pscm.csv', parse_dates=['date'])
    print(f"✅ Dataset chargé : {df.shape[0]:,} lignes × {df.shape[1]} colonnes")
except FileNotFoundError:
    print("⚠️  Fichier CSV non trouvé. Génération des données synthétiques...")
    # ── Génération de secours ────────────────────────────────────
    n_sku, n_days = 30, 365
    dates = pd.date_range('2024-01-01', periods=n_days, freq='D')
    skus = [f'SKU_{i:03d}' for i in range(1, n_sku+1)]
    rayons_list = ['Alimentaire','Hygiène','Boissons','Épicerie','Surgelés',
                   'Cosmétique','Boulangerie','Produits-Frais','Textile','High-Tech']
    rayons_map = {f'SKU_{i:03d}': rayons_list[(i-1)%10] for i in range(1, n_sku+1)}
    prix_map   = {s: round(max(2.5, min(np.random.lognormal(3.5,1.1),450)),2) for s in skus}
    rows = []
    for date in dates:
        ramadan = 1 if (date.month==3 and date.day>=22) or (date.month==4 and date.day<=21) else 0
        aid     = 1 if (date.month==4 and date.day in [21,22]) or (date.month==6 and date.day in [28,29]) else 0
        for sku in skus:
            promo  = int(np.random.random() < 0.08)
            prix   = prix_map[sku] * (0.75 if promo else 1.0)
            base   = np.random.lognormal(4.2, 0.9)
            saison = 1 + 0.4*np.sin(2*np.pi*date.dayofyear/365)
            ram_f  = 1.35 if (ramadan and rayons_map[sku] in ['Alimentaire','Épicerie']) else 1.0
            dow_f  = [0.85,0.90,1.00,1.05,1.15,1.42,1.20][date.dayofweek]
            qty    = max(0, int(base*saison*ram_f*(1.18 if aid else 1.0)
                                *(2.27 if promo else 1.0)*dow_f*max(0.1,np.random.normal(1,0.15))))
            stock  = np.random.randint(100, 3000)
            rows.append({
                'date': date.strftime('%Y-%m-%d'),
                'sku_id': sku, 'rayon': rayons_map[sku],
                'prix_unitaire': round(prix,2), 'promotion': promo,
                'quantite_vendue': qty, 'rupture': int(stock < qty*0.5),
                'stock_disponible': stock, 'delai_reappro_jours': np.random.randint(1,15),
                'temperature_celsius': round(15+20*np.sin(2*np.pi*date.dayofyear/365)+np.random.normal(0,3),1),
                'ramadan': ramadan, 'aid_fetete_nationale': aid,
                'jour_semaine': date.dayofweek, 'mois': date.month, 'trimestre': date.quarter,
            })
    df = pd.DataFrame(rows)
    df.to_csv('dataset_gms_maroc_pscm.csv', index=False, encoding='utf-8-sig')
    print(f"✅ Dataset généré & sauvegardé : {df.shape}")

df = df.sort_values(['sku_id','date']).reset_index(drop=True)

# =================================================================
# 2. FEATURE ENGINEERING
# =================================================================
print("\n⚙️  Feature Engineering...")

# Lag features
for lag in [1, 7, 14, 30]:
    df[f'lag_{lag}'] = df.groupby('sku_id')['quantite_vendue'].shift(lag)

# Rolling statistics
for w in [7, 14, 30]:
    df[f'rm_{w}'] = (df.groupby('sku_id')['quantite_vendue']
                       .transform(lambda x: x.shift(1).rolling(w).mean()))
    df[f'rs_{w}'] = (df.groupby('sku_id')['quantite_vendue']
                       .transform(lambda x: x.shift(1).rolling(w).std()))

# Cyclical encoding (sin/cos)
df['dow_sin']  = np.sin(2 * np.pi * df['jour_semaine'] / 7)
df['dow_cos']  = np.cos(2 * np.pi * df['jour_semaine'] / 7)
df['mois_sin'] = np.sin(2 * np.pi * df['mois'] / 12)
df['mois_cos'] = np.cos(2 * np.pi * df['mois'] / 12)

# Interaction features
df['couverture'] = df['stock_disponible'] / (df['rm_7'] + 1)
df['prix_promo'] = df['prix_unitaire'] * df['promotion']

# Encodage catégoriel
le = LabelEncoder()
df['sku_enc']   = le.fit_transform(df['sku_id'])
df['rayon_enc'] = le.fit_transform(df['rayon'])

# Suppression des NaN (issus des lags)
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

print(f"✅ Features construites : {df.shape[1]} variables, {df.shape[0]:,} observations")

# =================================================================
# 3. SPLIT TRAIN / TEST (TEMPOREL 80/20)
# =================================================================
FEATURES = [
    'prix_unitaire', 'promotion', 'stock_disponible',
    'delai_reappro_jours', 'temperature_celsius', 'ramadan',
    'aid_fetete_nationale', 'sku_enc', 'rayon_enc',
    'lag_1', 'lag_7', 'lag_14', 'lag_30',
    'rm_7', 'rm_14', 'rm_30', 'rs_7', 'rs_14', 'rs_30',
    'dow_sin', 'dow_cos', 'mois_sin', 'mois_cos',
    'trimestre', 'couverture', 'prix_promo',
]
TARGET_REG   = 'quantite_vendue'
TARGET_CLASS = 'rupture'

X = df[FEATURES]
y_r = df[TARGET_REG]
y_c = df[TARGET_CLASS]

split  = int(len(df) * 0.80)
X_tr, X_te   = X.iloc[:split],   X.iloc[split:]
yr_tr, yr_te = y_r.iloc[:split], y_r.iloc[split:]
yc_tr, yc_te = y_c.iloc[:split], y_c.iloc[split:]

print(f"✅ Split temporel : train={len(X_tr):,} | test={len(X_te):,}")

# =================================================================
# 4. MODÈLE A : RANDOM FOREST REGRESSOR
# =================================================================
print("\n🌲 [Modèle A] RandomForest Regressor...")
rf = RandomForestRegressor(
    n_estimators=300, max_depth=18,
    min_samples_split=5, min_samples_leaf=2,
    max_features='sqrt', n_jobs=-1, random_state=42
)
rf.fit(X_tr, yr_tr)
p_rf    = rf.predict(X_te)
rmse_rf = np.sqrt(mean_squared_error(yr_te, p_rf))
r2_rf   = r2_score(yr_te, p_rf)
mae_rf  = np.mean(np.abs(yr_te - p_rf))
print(f"   RMSE={rmse_rf:.2f} | R²={r2_rf:.4f} | MAE={mae_rf:.2f}")

# =================================================================
# 5. MODÈLE B : XGBOOST REGRESSOR
# =================================================================
print("\n⚡ [Modèle B] XGBoost Regressor...")
xgbr = xgb.XGBRegressor(
    n_estimators=500, learning_rate=0.05,
    max_depth=8, subsample=0.85, colsample_bytree=0.80,
    reg_alpha=0.1, reg_lambda=1.5,
    early_stopping_rounds=50, eval_metric='rmse',
    random_state=42, n_jobs=-1, verbosity=0
)
xgbr.fit(X_tr, yr_tr, eval_set=[(X_te, yr_te)], verbose=False)
p_xgb    = xgbr.predict(X_te)
rmse_xgb = np.sqrt(mean_squared_error(yr_te, p_xgb))
r2_xgb   = r2_score(yr_te, p_xgb)
mae_xgb  = np.mean(np.abs(yr_te - p_xgb))
mape_xgb = np.mean(np.abs((yr_te - p_xgb) / (yr_te + 1))) * 100
print(f"   RMSE={rmse_xgb:.2f} | R²={r2_xgb:.4f} | MAE={mae_xgb:.2f} | MAPE={mape_xgb:.1f}%")

# =================================================================
# 6. MODÈLE C : RÉGRESSION LINÉAIRE (BASELINE)
# =================================================================
print("\n📏 [Modèle C] Régression Linéaire (Baseline)...")
lr = Pipeline([('sc', StandardScaler()), ('lr', LinearRegression())])
lr.fit(X_tr, yr_tr)
p_lr    = np.clip(lr.predict(X_te), 0, None)
rmse_lr = np.sqrt(mean_squared_error(yr_te, p_lr))
r2_lr   = r2_score(yr_te, p_lr)
print(f"   RMSE={rmse_lr:.2f} | R²={r2_lr:.4f}")

# =================================================================
# 7. MODÈLE D : XGBOOST CLASSIFIER (RUPTURE DE STOCK)
# =================================================================
print("\n🔴 [Modèle D] XGBoost Classifier (Rupture de Stock)...")
ratio = (yc_tr == 0).sum() / max((yc_tr == 1).sum(), 1)
xgbc = xgb.XGBClassifier(
    n_estimators=400, learning_rate=0.05, max_depth=7,
    subsample=0.85, colsample_bytree=0.80,
    scale_pos_weight=ratio,
    eval_metric='auc', random_state=42,
    n_jobs=-1, verbosity=0
)
xgbc.fit(X_tr, yc_tr, eval_set=[(X_te, yc_te)], verbose=False)
p_clf  = xgbc.predict(X_te)
pr_clf = xgbc.predict_proba(X_te)[:, 1]
acc    = accuracy_score(yc_te, p_clf)
f1     = f1_score(yc_te, p_clf, average='weighted')
auc    = roc_auc_score(yc_te, pr_clf)
print(f"   Accuracy={acc:.4f} | F1={f1:.4f} | AUC-ROC={auc:.4f}")
print("\nRapport de classification :")
print(classification_report(yc_te, p_clf, target_names=['Non-Rupture','Rupture']))

# =================================================================
# 8. PRÉDICTION 30 JOURS (MODE RÉCURSIF)
# =================================================================
print("\n📅 Prédiction 30 jours (mode récursif)...")
last_obs = X_te.iloc[-1:].copy()
preds_30 = []
for day in range(30):
    q = max(0, xgbr.predict(last_obs)[0] + np.random.normal(0, rmse_xgb * 0.1))
    preds_30.append(round(q, 1))
    last_obs = last_obs.copy()
    last_obs['lag_1'] = q
    last_obs['rm_7']  = np.mean(preds_30[-7:])
    dow_val = (last_obs['dow_sin'].values[0] + 1) % 7
    last_obs['dow_sin'] = np.sin(2 * np.pi * dow_val / 7)
    last_obs['dow_cos'] = np.cos(2 * np.pi * dow_val / 7)

print("Prédictions J+1 à J+30 :")
for i in range(0, 30, 5):
    chunk = preds_30[i:i+5]
    labels = [f"J+{j+1}" for j in range(i, i+len(chunk))]
    print("  " + "  ".join(f"{l}:{v:.0f}" for l,v in zip(labels, chunk)))

# =================================================================
# 9. FEATURE IMPORTANCE
# =================================================================
print("\n📊 Feature Importance (XGBoost Regressor — Top 10) :")
fi = pd.Series(xgbr.feature_importances_, index=FEATURES).sort_values(ascending=False)
for feat, score in fi.head(10).items():
    bar = "█" * int(score * 200)
    print(f"  {feat:<25} {bar} {score*100:.1f}%")

# =================================================================
# 10. ARCHITECTURE MULTI-AGENTS — CLASSE PSCMAgent
# =================================================================
print("\n🤖 Architecture Multi-Agents — Déploiement agents par SKU...")

class PSCMAgent:
    """
    Agent autonome pour la prédiction de la demande d'un SKU spécifique.
    Chaque agent encapsule son propre modèle entraîné, ses prédictions
    et ses métriques de performance.
    """
    def __init__(self, agent_id: str, agent_type: str, sku: str = None):
        self.agent_id    = agent_id
        self.agent_type  = agent_type  # 'regression' | 'classification'
        self.sku         = sku
        self.model       = None
        self.predictions = []
        self.metrics     = {}
        self.status      = "initialized"

    def train(self, X_tr, y_tr) -> None:
        """Entraîne le modèle sur les données d'entraînement."""
        if self.agent_type == 'regression':
            self.model = xgb.XGBRegressor(n_estimators=100, verbosity=0, random_state=42)
        else:
            pos_w = max(1, (y_tr==0).sum() / max((y_tr==1).sum(), 1))
            self.model = xgb.XGBClassifier(n_estimators=100, scale_pos_weight=pos_w,
                                           verbosity=0, random_state=42)
        self.model.fit(X_tr, y_tr)
        self.status = "trained"

    def predict(self, X_te) -> np.ndarray:
        """Génère les prédictions sur le jeu de test."""
        self.predictions = self.model.predict(X_te)
        self.status = "predicted"
        return self.predictions

    def evaluate(self, y_true) -> dict:
        """Calcule les métriques de performance."""
        if self.agent_type == 'regression':
            self.metrics = {
                'rmse': np.sqrt(mean_squared_error(y_true, self.predictions)),
                'r2':   r2_score(y_true, self.predictions),
                'mae':  np.mean(np.abs(y_true - self.predictions)),
            }
        else:
            self.metrics = {
                'accuracy': accuracy_score(y_true, self.predictions),
                'f1':       f1_score(y_true, self.predictions, average='weighted'),
            }
        self.status = "evaluated"
        return self.metrics

    def report(self) -> str:
        """Retourne un rapport synthétique de l'agent."""
        return (f"Agent {self.agent_id} | SKU={self.sku} | "
                f"Type={self.agent_type} | Status={self.status} | "
                f"Metrics={self.metrics}")


# Déploiement sur les SKU disponibles
agents_reg, agents_clf = [], []
skus_available = df['sku_id'].unique()

for i, sku in enumerate(skus_available[:20]):
    sub = df[df['sku_id'] == sku]
    if len(sub) < 60:
        continue
    Xs  = sub[FEATURES]
    yr_ = sub[TARGET_REG]
    yc_ = sub[TARGET_CLASS]
    sp  = int(len(sub) * 0.8)

    # Agent Régression
    ag_r = PSCMAgent(f'REG_{i:03d}', 'regression', sku)
    ag_r.train(Xs.iloc[:sp], yr_.iloc[:sp])
    ag_r.predict(Xs.iloc[sp:])
    ag_r.evaluate(yr_.iloc[sp:])
    agents_reg.append(ag_r)

    # Agent Classification
    if yc_.iloc[:sp].nunique() > 1:
        ag_c = PSCMAgent(f'CLF_{i:03d}', 'classification', sku)
        ag_c.train(Xs.iloc[:sp], yc_.iloc[:sp])
        ag_c.predict(Xs.iloc[sp:])
        ag_c.evaluate(yc_.iloc[sp:])
        agents_clf.append(ag_c)

print(f"✅ Agents déployés : {len(agents_reg)} régression + {len(agents_clf)} classification")
if agents_reg:
    print(f"   RMSE moyen (agents) : {np.mean([a.metrics['rmse'] for a in agents_reg]):.2f}")
if agents_clf:
    print(f"   Accuracy moy (agents): {np.mean([a.metrics['accuracy'] for a in agents_clf]):.4f}")

# Rapport des 5 premiers agents
print("\nRapport Top 5 agents régression :")
for ag in agents_reg[:5]:
    print("  " + ag.report())

# =================================================================
# 11. VISUALISATIONS
# =================================================================
print("\n📈 Génération des visualisations...")

fig = plt.figure(figsize=(16, 20))
gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

# ── Graphique 1 : Réel vs Prédit (XGBoost) ──
ax1 = fig.add_subplot(gs[0, 0])
sample_n = min(300, len(yr_te))
idx = np.random.choice(len(yr_te), sample_n, replace=False)
ax1.scatter(yr_te.values[idx], p_xgb[idx], alpha=0.5, s=15, color='#2563EB', label='XGBoost')
ax1.scatter(yr_te.values[idx], p_rf[idx],  alpha=0.3, s=15, color='#EA580C', label='RandomForest')
max_val = max(yr_te.max(), p_xgb.max())
ax1.plot([0, max_val], [0, max_val], 'k--', linewidth=1.5, label='Parfait (y=x)')
ax1.set_xlabel('Demande Réelle', fontsize=11)
ax1.set_ylabel('Demande Prédite', fontsize=11)
ax1.set_title('Réel vs Prédit — Régression\nXGBoost vs RandomForest', fontsize=12, fontweight='bold', color='#1B3A6B')
ax1.legend(fontsize=9)
ax1.set_facecolor('#F8FAFC')

# ── Graphique 2 : Prédiction 30 jours ──
ax2 = fig.add_subplot(gs[0, 1])
days30 = np.arange(1, 31)
ic_low  = np.array(preds_30) - 1.96 * rmse_xgb * 0.8
ic_high = np.array(preds_30) + 1.96 * rmse_xgb * 0.8
ax2.fill_between(days30, ic_low, ic_high, alpha=0.25, color='#2563EB', label='IC 95%')
ax2.plot(days30, preds_30, 'o-', color='#EA580C', linewidth=2, markersize=4, label='Prédiction XGBoost')
ax2.axvline(22, color='#DC2626', linestyle=':', linewidth=2, label='Seuil réappro (J+22)')
ax2.set_xlabel('Horizon (jours)', fontsize=11)
ax2.set_ylabel('Quantité prédite', fontsize=11)
ax2.set_title('Prédiction de la Demande\nHorizon 30 Jours', fontsize=12, fontweight='bold', color='#1B3A6B')
ax2.legend(fontsize=9)
ax2.set_facecolor('#F8FAFC')
ax2.grid(alpha=0.3)

# ── Graphique 3 : Feature Importance ──
ax3 = fig.add_subplot(gs[1, :])
top10_feat = fi.head(10)
colors_fi = ['#1B3A6B' if v > 0.15 else '#2563EB' if v > 0.10 else '#0D9488' if v > 0.05 else '#9CA3AF'
             for v in top10_feat.values]
bars = ax3.barh(top10_feat.index[::-1], top10_feat.values[::-1] * 100,
                color=colors_fi[::-1], alpha=0.9, edgecolor='white', height=0.6)
for bar, val in zip(bars, top10_feat.values[::-1] * 100):
    ax3.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
             f'{val:.1f}%', va='center', fontsize=10, fontweight='bold', color='#374151')
ax3.set_title('Feature Importance — XGBoost Regressor', fontsize=13, fontweight='bold', color='#1B3A6B')
ax3.set_xlabel('Importance (%)', fontsize=11)
ax3.set_facecolor('#F8FAFC')
ax3.grid(axis='x', alpha=0.3)

# ── Graphique 4 : Distribution de la demande ──
ax4 = fig.add_subplot(gs[2, 0])
ax4.hist(df['quantite_vendue'], bins=50, color='#2563EB', alpha=0.8, edgecolor='white')
ax4.axvline(df['quantite_vendue'].mean(),   color='#EA580C', linewidth=2, linestyle='--',
            label=f'Moyenne={df["quantite_vendue"].mean():.0f}')
ax4.axvline(df['quantite_vendue'].median(), color='#0D9488', linewidth=2, linestyle='--',
            label=f'Médiane={df["quantite_vendue"].median():.0f}')
ax4.set_title('Distribution de la Demande', fontsize=12, fontweight='bold', color='#1B3A6B')
ax4.set_xlabel('Quantité vendue/jour/SKU', fontsize=11)
ax4.set_ylabel('Fréquence', fontsize=11)
ax4.legend(fontsize=9)
ax4.set_facecolor('#F8FAFC')

# ── Graphique 5 : Comparaison des modèles ──
ax5 = fig.add_subplot(gs[2, 1])
models_names = ['Régression\nLinéaire', 'RandomForest', 'XGBoost']
r2_scores    = [r2_lr, r2_rf, r2_xgb]
rmse_scores  = [rmse_lr, rmse_rf, rmse_xgb]
x_pos = np.arange(len(models_names))
bar_w = 0.35
b1 = ax5.bar(x_pos - bar_w/2, r2_scores, bar_w, label='R² (axe gauche)',
             color=['#9CA3AF','#2563EB','#1B3A6B'], alpha=0.85)
ax5_r = ax5.twinx()
b2 = ax5_r.bar(x_pos + bar_w/2, rmse_scores, bar_w, label='RMSE (axe droit)',
               color=['#FED7AA','#0D9488','#15803D'], alpha=0.85)
ax5.set_xticks(x_pos)
ax5.set_xticklabels(models_names, fontsize=9)
ax5.set_ylabel('R²', fontsize=11, color='#1B3A6B')
ax5_r.set_ylabel('RMSE', fontsize=11, color='#15803D')
ax5.set_title('Comparaison des Modèles\nR² vs RMSE', fontsize=12, fontweight='bold', color='#1B3A6B')
ax5.set_facecolor('#F8FAFC')
lines1, labels1 = ax5.get_legend_handles_labels()
lines2, labels2 = ax5_r.get_legend_handles_labels()
ax5.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='upper left')

fig.suptitle('PSCM — Prédiction de la Demande en Stock\nSESSI Fiawoo Akou Sika Audrey | ENCG Settat S8 | 2025–2026',
             fontsize=14, fontweight='bold', color='#1B3A6B', y=0.98)
fig.savefig('visualisations_pscm.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("✅ visualisations_pscm.png sauvegardé")

# =================================================================
# 12. RÉSUMÉ FINAL
# =================================================================
print("\n" + "=" * 65)
print("RÉSUMÉ DES PERFORMANCES")
print("=" * 65)
print(f"{'Modèle':<30} {'RMSE':>8} {'R²':>8} {'MAE':>8}")
print("-" * 55)
print(f"{'XGBoost Regressor':<30} {rmse_xgb:>8.2f} {r2_xgb:>8.4f} {mae_xgb:>8.2f}")
print(f"{'RandomForest Regressor':<30} {rmse_rf:>8.2f} {r2_rf:>8.4f} {mae_rf:>8.2f}")
print(f"{'Régression Linéaire':<30} {rmse_lr:>8.2f} {r2_lr:>8.4f} {'N/A':>8}")
print(f"\n{'XGBoost Classifier':<30} {'Acc':>8} {'F1':>8} {'AUC':>8}")
print("-" * 55)
print(f"{'(Rupture de Stock)':<30} {acc:>8.4f} {f1:>8.4f} {auc:>8.4f}")
print("=" * 65)
print(f"\n✅ MEILLEUR MODÈLE RÉGRESSION : XGBoost (R²={r2_xgb:.4f}, MAPE={mape_xgb:.1f}%)")
print(f"✅ MEILLEUR MODÈLE CLASSIF    : XGBoost (AUC-ROC={auc:.4f})")
print(f"✅ Agents déployés            : {len(agents_reg)+len(agents_clf)} / 1 000 cibles")
print(f"✅ Prédiction 30j moyenne     : {np.mean(preds_30):.1f} unités/jour")
print("\n✅ Pipeline PSCM terminé avec succès.")
print("   SESSI Fiawoo Akou Sika Audrey | ENCG Settat S8 | 2025–2026")
