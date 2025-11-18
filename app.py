# app.py
import math
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import t
import shap
import matplotlib.pyplot as plt

# -----------------------------
# Utility / Config
# -----------------------------
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

@dataclass
class NNConfig:
    input_dim: int
    depth: int
    width: int
    dropout: float = 0.05

def suggest_arch(n_samples: int, p: int) -> NNConfig:
    depth = int(np.clip(np.floor(np.log10(max(n_samples, 50))), 1, 4))
    width = int(np.clip(8 * p * depth, 16, 512))
    return NNConfig(input_dim=p, depth=depth, width=width, dropout=0.05)

class MLP(nn.Module):
    def __init__(self, cfg: NNConfig):
        super().__init__()
        layers: List[nn.Module] = []
        in_dim = cfg.input_dim
        for _ in range(cfg.depth):
            layers += [
                nn.Linear(in_dim, cfg.width),
                nn.ReLU(),
                nn.Dropout(cfg.dropout),
            ]
            in_dim = cfg.width
        layers += [nn.Linear(in_dim, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)

def train_model(
    X: np.ndarray,
    y: np.ndarray,
    *,
    epochs: int = 500,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 256,
    patience: int = 20,
    verbose: bool = True,
    cfg_override: Optional[NNConfig] = None,   # ← 追加：手動アーキテクチャ
) -> Tuple[MLP, float, NNConfig]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=min(0.2, max(0.1, 1000 / max(1, len(X)))), random_state=SEED
    )

    cfg = cfg_override if cfg_override is not None else suggest_arch(n_samples=X_tr.shape[0], p=X.shape[1])
    model = MLP(cfg).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    def to_tensor(a): return torch.tensor(a, dtype=torch.float32, device=device)
    X_tr_t, y_tr_t = to_tensor(X_tr), to_tensor(y_tr)
    X_val_t, y_val_t = to_tensor(X_val), to_tensor(y_val)

    best_val = float("inf")
    best_state = None
    wait = 0
    n_batches = max(1, math.ceil(len(X_tr_t) / batch_size))

    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(X_tr_t), device=device)
        tr_loss_acc = 0.0

        for b in range(n_batches):
            idx = perm[b * batch_size : (b + 1) * batch_size]
            xb, yb = X_tr_t[idx], y_tr_t[idx]
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            tr_loss_acc += loss.item() * len(idx)

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = loss_fn(val_pred, y_val_t).item()

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = model.state_dict()
            wait = 0
        else:
            wait += 1

        if verbose and ep % 25 == 0:
            st.write(f"Epoch {ep:4d} | train_loss≈{tr_loss_acc/len(X_tr_t):.5f} | val_loss={val_loss:.5f}")

        if wait >= patience:
            if verbose:
                st.write(f"Early stopping at epoch {ep}. Best val_loss={best_val:.5f}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val, cfg

def predict(model: MLP, X: np.ndarray) -> np.ndarray:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    with torch.no_grad():
        x = torch.tensor(X, dtype=torch.float32, device=device)
        yhat = model(x).cpu().numpy()
    return yhat

# -----------------------------
# Streamlit App
# -----------------------------
st.set_page_config(page_title="Causal NN (ATE / CATE / SHAP)", layout="wide")

st.title("簡易因果推論ツール（T-learner＋共変量バランス・重要度/SHAP）")

with st.sidebar:
    st.header("学習ハイパーパラメータ")
    epochs = st.number_input("最大エポック数", min_value=50, max_value=5000, value=500, step=50)
    lr = st.number_input("学習率 (Adam)", min_value=1e-5, max_value=1e-1, value=1e-3, step=1e-5, format="%.5f")
    weight_decay = st.number_input("Weight Decay", min_value=0.0, max_value=1e-1, value=1e-4, step=1e-5, format="%.5f")
    batch_size = st.number_input("バッチサイズ", min_value=16, max_value=8192, value=256, step=16)
    patience = st.number_input("EarlyStopping patience", min_value=5, max_value=200, value=20, step=5)
    st.markdown("---")
    st.caption("ネット深さ・幅は各群のサンプルサイズと特徴量次元から自動提案します。")

st.subheader("1) データのアップロード")
file = st.file_uploader("CSV を選択してください（UTF-8推奨）", type=["csv"])

if file is not None:
    df = pd.read_csv(file)
    st.write("データプレビュー（先頭）")
    st.dataframe(df.head())

    with st.expander("列の指定", expanded=True):
        all_cols = list(df.columns)
        id_col = st.selectbox("ID 列（任意）", options=["(なし)"] + all_cols, index=0)
        D_col = st.selectbox("処置フラグ列 (0/1)", options=all_cols)
        Y_col = st.selectbox("結果変数 Y 列（連続）", options=[c for c in all_cols if c != D_col])
        X_cols = st.multiselect(
            "共変量 X の列（1列以上）",
            options=[c for c in all_cols if c not in {D_col, Y_col}],
            default=[c for c in all_cols if c not in {D_col, Y_col}][:min(5, max(1, len(all_cols) - 2))]
        )

    if len(X_cols) == 0:
        st.warning("共変量 X を1列以上選んでください。")
        st.stop()

    use_cols = ([id_col] if id_col != "(なし)" else []) + [D_col, Y_col] + X_cols
    work = df[use_cols].copy()
    n0_all = len(work)
    work = work.dropna()
    if len(work) < n0_all:
        st.info(f"欠損を含む {n0_all - len(work)} 行を除外しました。")

    try:
        D = work[D_col].astype(int).values
        y = work[Y_col].astype(float).values
        X = work[X_cols].astype(float).values
    except Exception as e:
        st.error(f"数値変換に失敗しました。列の型をご確認ください。詳細: {e}")
        st.stop()

    if not set(np.unique(D)).issubset({0, 1}):
        st.error("処置フラグ列 D は 0/1 である必要があります。")
        st.stop()

    # ---- Sidebar: 推奨アーキテクチャ + 上書きUI（要求 3 & 2）----
    mask1_sidebar = (D == 1)
    mask0_sidebar = (D == 0)
    n1_sidebar, n0_sidebar = mask1_sidebar.sum(), mask0_sidebar.sum()
    p_sidebar = X.shape[1]
    sugg1 = suggest_arch(n_samples=n1_sidebar, p=p_sidebar)
    sugg0 = suggest_arch(n_samples=n0_sidebar, p=p_sidebar)

    with st.sidebar:
        st.subheader("自動推奨アーキテクチャ")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**処置群 (D=1)**")
            st.write(f"推奨 depth = {sugg1.depth}")
            st.write(f"推奨 width  = {sugg1.width}")
        with c2:
            st.markdown("**対照群 (D=0)**")
            st.write(f"推奨 depth = {sugg0.depth}")
            st.write(f"推奨 width  = {sugg0.width}")

        st.markdown("---")
        st.markdown("**手動で上書き（任意）**")
        override = st.checkbox("推奨の深さ・幅を上書きする", value=False)
        if override:
            depth1 = st.number_input("処置群 depth", min_value=1, max_value=8, value=sugg1.depth, step=1, key="ov_d1")
            width1  = st.number_input("処置群 width",  min_value=8, max_value=2048, value=sugg1.width, step=8, key="ov_w1")
            depth0 = st.number_input("対照群 depth", min_value=1, max_value=8, value=sugg0.depth, step=1, key="ov_d0")
            width0  = st.number_input("対照群 width",  min_value=8, max_value=2048, value=sugg0.width, step=8, key="ov_w0")
        else:
            depth1, width1 = sugg1.depth, sugg1.width
            depth0, width0 = sugg0.depth, sugg0.width

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    mask1 = (D == 1)
    mask0 = (D == 0)
    n1, n0 = mask1.sum(), mask0.sum()

    # ---- (要求1) 共変量バランス：箱ひげ図（均等サイズ & 凡例説明）----
    st.subheader("1.5) 共変量バランス（処置群 vs 対照群）")
    st.caption("処置群 (D=1) と対照群 (D=0) で各共変量の分布を箱ひげ図で比較します。")
    st.caption("※ 緑色の▲が平均値、オレンジ色の横線が中央値です。")
    vars_to_plot = st.multiselect(
        "箱ひげ図で表示する共変量（複数選択）",
        options=X_cols,
        default=X_cols[:min(6, len(X_cols))]
    )
    if len(vars_to_plot) > 0:
        ncols = 3
        rows = [vars_to_plot[i:i+ncols] for i in range(0, len(vars_to_plot), ncols)]
        for row_vars in rows:
            cols_fig = st.columns(ncols)  # ← 常に同じ列数で均等幅
            for ci in range(ncols):
                with cols_fig[ci]:
                    if ci < len(row_vars):
                        var = row_vars[ci]
                        fig, ax = plt.subplots(figsize=(4, 3))
                        data_treat = work.loc[mask1, var].values
                        data_ctrl  = work.loc[mask0, var].values
                        meanprops = dict(marker='^', markeredgecolor='green', markerfacecolor='green')
                        medianprops = dict(color='orange', linewidth=2)
                        ax.boxplot([data_ctrl, data_treat],
                                   labels=["D=0", "D=1"],
                                   showmeans=True,
                                   meanprops=meanprops,
                                   medianprops=medianprops)
                        ax.set_title(var)
                        st.pyplot(fig)
                        plt.close(fig)
                    else:
                        st.empty()

    # ---- Step 2: 学習 ----
    st.subheader("2) 学習の実行")
    st.write(f"処置群 n1 = {n1}、対照群 n0 = {n0}、特徴量次元 p = {Xs.shape[1]}")
    insufficient = []
    if n1 < 10 or n1 < Xs.shape[1] + 5:
        insufficient.append("処置群")
    if n0 < 10 or n0 < Xs.shape[1] + 5:
        insufficient.append("対照群")
    if insufficient:
        st.warning(" / ".join(insufficient) + " のサンプルが少なく過学習しやすい可能性があります。結果の解釈に注意してください。")

    if st.button("学習開始 / 再学習", type="primary"):
        start = time.time()
        cfg1 = NNConfig(input_dim=Xs.shape[1], depth=int(depth1), width=int(width1), dropout=0.05)
        cfg0 = NNConfig(input_dim=Xs.shape[1], depth=int(depth0), width=int(width0), dropout=0.05)

        with st.spinner("処置群モデル f1 を学習中..."):
            f1, val1, used_cfg1 = train_model(
                Xs[mask1], y[mask1],
                epochs=int(epochs), lr=float(lr), weight_decay=float(weight_decay),
                batch_size=int(batch_size), patience=int(patience), verbose=True,
                cfg_override=cfg1
            )
        with st.spinner("対照群モデル f0 を学習中..."):
            f0, val0, used_cfg0 = train_model(
                Xs[mask0], y[mask0],
                epochs=int(epochs), lr=float(lr), weight_decay=float(weight_decay),
                batch_size=int(batch_size), patience=int(patience), verbose=True,
                cfg_override=cfg0
            )

        with st.spinner("ATE を計算中..."):
            mu1_hat = predict(f1, Xs)
            mu0_hat = predict(f0, Xs)
            tau_hat = mu1_hat - mu0_hat
            ate_hat = float(np.mean(tau_hat))

            n = len(tau_hat)
            if n <= 1:
                st.warning("サンプル数が少なすぎるため信頼区間を計算できません。")
                se_hat = np.nan
                ci_low, ci_high = np.nan, np.nan
            else:
                var_hat = np.var(tau_hat, ddof=1) / n
                se_hat = float(np.sqrt(var_hat))
                df_ = n - 1
                t_crit = float(t.ppf(0.975, df_))
                ci_low = ate_hat - t_crit * se_hat
                ci_high = ate_hat + t_crit * se_hat

        # セッション保存（永続表示用）
        st.session_state["scaler"] = scaler
        st.session_state["f1"] = f1
        st.session_state["f0"] = f0
        st.session_state["X_cols"] = X_cols
        st.session_state["mu1_hat"] = mu1_hat
        st.session_state["mu0_hat"] = mu0_hat
        st.session_state["tau_hat"] = tau_hat
        st.session_state["ate_hat"] = ate_hat
        st.session_state["se_hat"] = se_hat
        st.session_state["ci"] = (ci_low, ci_high)
        st.session_state["val1"] = val1
        st.session_state["val0"] = val0
        st.session_state["X"] = X
        st.session_state["cfg1_used"] = used_cfg1
        st.session_state["cfg0_used"] = used_cfg0
        st.session_state["work_df"] = work  # DL用

        elapsed = time.time() - start
        st.success(f"学習完了（{elapsed:.1f} 秒）")

    # ---- 直前の学習結果（常時表示） + ダウンロードも常時表示（要求 3）----
    if "ate_hat" in st.session_state:
        st.subheader("直前の学習結果")
        st.metric("推定 ATE", f"{st.session_state['ate_hat']:.6f}")
        st.write(f"標準誤差 (SE) = {st.session_state['se_hat']:.6f}")
        ci_low, ci_high = st.session_state["ci"]
        st.write(f"95% 信頼区間: [{ci_low:.6f}, {ci_high:.6f}]")
        with st.expander("開発メモ（前回学習のバリデーション MSE / 使用アーキ）"):
            st.write(f"f1 val_MSE ≈ {st.session_state['val1']:.6f} / f0 val_MSE ≈ {st.session_state['val0']:.6f}")
            c1, c0 = st.session_state.get("cfg1_used"), st.session_state.get("cfg0_used")
            if c1 and c0:
                st.write(f"f1 used depth={c1.depth}, width={c1.width} | f0 used depth={c0.depth}, width={c0.width}")

        # 常時ダウンロード可能
        if all(k in st.session_state for k in ["mu1_hat", "mu0_hat", "tau_hat", "work_df"]):
            out = st.session_state["work_df"].copy()
            out["f1_hat"] = st.session_state["mu1_hat"]
            out["f0_hat"] = st.session_state["mu0_hat"]
            out["cate_hat"] = st.session_state["tau_hat"]
            csv = out.to_csv(index=False).encode("utf-8")
            st.download_button(
                "個別の予測値（f1, f0, cate）をCSVでダウンロード",
                data=csv,
                file_name="predictions_with_CATE.csv",
                mime="text/csv",
            )

    # ---- Step 3: 任意の X で CATE ----
    st.subheader("3) 任意の X で CTE/CATE を計算")
    if "f1" in st.session_state and "f0" in st.session_state:
        st.caption("下で各特徴量の値を指定すると、f1(x), f0(x), CATE(x) を返します。")
        user_x = []
        cols_per_row = 3
        cols_list = st.session_state["X_cols"]
        means = pd.DataFrame(X, columns=cols_list).mean()

        rows = [cols_list[i : i + cols_per_row] for i in range(0, len(cols_list), cols_per_row)]
        for row in rows:
            col_widgets = st.columns(len(row))
            for j, colname in enumerate(row):
                default_val = float(means[colname])
                v = col_widgets[j].number_input(f"{colname}", value=default_val)
                user_x.append(v)

        if st.button("この X で CATE を計算", type="secondary"):
            x_arr = np.array(user_x, dtype=float).reshape(1, -1)
            xs = st.session_state["scaler"].transform(x_arr)
            f1x = float(predict(st.session_state["f1"], xs)[0])
            f0x = float(predict(st.session_state["f0"], xs)[0])
            catex = f1x - f0x
            st.write(f"**f1(x)** = {f1x:.6f} / **f0(x)** = {f0x:.6f}")
            st.metric("CATE(x) = f1(x) - f0(x)", f"{catex:.6f}")
    else:
        st.info("まず学習を実行してください。")

    # ---- Step 4: CATE ドライバー分析（RF 重要度 → 先に表示 / SHAP は後で実行）----
    st.subheader("4) 不均一処置効果（CATE）のドライバー分析：Random Forest 重要度 ＋ SHAP")
    if "tau_hat" in st.session_state:
        tau_hat = st.session_state["tau_hat"]
        X_rf = st.session_state.get("X", X)

        rf_n_estimators = st.number_input("RandomForest 木の本数 (n_estimators)", min_value=50, max_value=2000, value=500, step=50)
        rf_max_depth = st.number_input("最大深さ (None=0)", min_value=0, max_value=50, value=0, step=1)
        rf = RandomForestRegressor(
            n_estimators=int(rf_n_estimators),
            max_depth=None if rf_max_depth == 0 else int(rf_max_depth),
            random_state=SEED,
            n_jobs=-1,
        )
        with st.spinner("RandomForest を学習中..."):
            rf.fit(X_rf, tau_hat)

        # 重要度（先に即表示）
        importances = rf.feature_importances_
        imp_df = pd.DataFrame({"feature": st.session_state["X_cols"], "importance": importances})
        imp_df = imp_df.sort_values("importance", ascending=False).reset_index(drop=True)

        st.markdown("**特徴量重要度（RandomForest）**")
        fig_imp, ax_imp = plt.subplots(figsize=(6, max(3, 0.35*len(imp_df))))
        ax_imp.barh(imp_df["feature"][::-1], imp_df["importance"][::-1])
        ax_imp.set_xlabel("importance")
        ax_imp.set_ylabel("feature")
        for i, v in enumerate(imp_df["importance"][::-1]):
            ax_imp.text(v, i, f"{v:.3f}", va="center", ha="left")
        plt.tight_layout()
        st.pyplot(fig_imp)
        plt.close(fig_imp)

        st.dataframe(imp_df)

        # SHAP は重いので、ユーザー操作で実行（棒グラフを先に見られるようにする）
        #if st.button("SHAP を計算して表示（Beeswarm 図） ※最大数分かかります"):
        st.caption("SHAP を計算中...")
        with st.spinner("SHAP を計算中..."):
            n_shap = min(5000, X_rf.shape[0])
            rng = np.random.default_rng(SEED)
            idx = rng.choice(X_rf.shape[0], size=n_shap, replace=False) if X_rf.shape[0] > n_shap else np.arange(X_rf.shape[0])
            X_shap = X_rf[idx]

            explainer = shap.TreeExplainer(rf)
            shap_values = explainer.shap_values(X_shap)  # (n, p)

        st.markdown("**SHAP Beeswarm（各特徴量の値の大小と CATE への影響方向 ±）**")
        fig_swarm = plt.figure(figsize=(7, 4 + 0.15*len(st.session_state["X_cols"])))
        shap.summary_plot(shap_values, features=X_shap, feature_names=st.session_state["X_cols"], show=False)
        st.pyplot(fig_swarm)
        plt.close(fig_swarm)
        st.caption("注：色は特徴量値の大小（右側のカラーバー参照）、左右は CATE への影響方向（±）を表します。")
    else:
        st.info("学習後に CATE（個体別効果）が得られると、本セクションが有効になります。")

else:
    st.info("CSV をアップロードすると列選択と学習が可能になります。")

st.markdown("---")
st.caption(
    "注意: これは教育・検証目的のデモです。推定はモデル化仮定に依存し、未観測交絡・選択バイアス・外挿のリスクに留意してください。"
)
