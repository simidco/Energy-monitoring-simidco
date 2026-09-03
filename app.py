# ===============================================
# داشبورد پایش برق کنسانتره - نسخه تجهیز‌محور واحد
# ===============================================
# -------------------- Import Libraries --------------------
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from persiantools.jdatetime import JalaliDate
import os
import io
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import IsolationForest
import statsmodels.api as sm
import pulp
from pulp import LpProblem, LpMinimize, LpMaximize, LpVariable, LpStatus, value
from scipy.optimize import minimize
# ReportLab
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Image, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
import plotly.io as pio
import plotly.graph_objects as go
import hashlib
import urllib.parse
import hmac
import streamlit as st
import statsmodels.api as sm

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

def compute_official_regression_baseline(df_consumption, extra_vars_df=None,
                                         manual_x=None, disable_filters=False,
                                         baseline_method="ثابت"):
    """
    محاسبه Baseline با رگرسیون چندمتغیره (مطابق اکسل)
    - manual_x: اگر کاربر یک ستون خاص انتخاب کند، فقط همان ستون به‌عنوان متغیر مستقل در نظر گرفته می‌شود.
    - disable_filters: غیرفعال‌سازی شرط‌های P-Value و R²
    - baseline_method: "ثابت" یا "متغیر"
    برمی‌گرداند: (baseline_value, diagnostics) که diagnostics شامل predictions و residuals نیز هست.
    """
    diagnostics = {
        "notes": [], "selected_vars": [], "r2": None, "equation": "",
        "method": "regression_official", "status": "success",
        "intercept": None, "coef": None, "n_points": None,
        "baseline_per_point": None, "vif": {},
        "predictions": [],    # جدید
        "residuals": [],      # جدید
        "actual": []          # جدید
    }
    try:
        df = df_consumption.copy().dropna(subset=["consumption"])
        
        # حداقل نقاط
        min_points = 6 if len(df) <= 60 else 10
        if len(df) < min_points:
            diagnostics["notes"].append(f"داده ناکافی ({len(df)} نقطه)")
            diagnostics["status"] = "insufficient_data"
            fallback_val = df["consumption"].median() if not df.empty else 0
            return round(fallback_val, 3), diagnostics
        
        # حذف پرت‌ها (IQR)
        Q1 = df["consumption"].quantile(0.25)
        Q3 = df["consumption"].quantile(0.75)
        IQR = Q3 - Q1
        lower = max(0, Q1 - 1.5 * IQR)
        upper = Q3 + 1.5 * IQR
        df_clean = df[(df["consumption"] >= lower) & (df["consumption"] <= upper)].copy()
        removed = len(df) - len(df_clean)
        if removed > 0:
            diagnostics["notes"].append(f"حذف {removed} پرت")
        
        if len(df_clean) < 5:
            diagnostics["notes"].append("داده کافی نماند")
            diagnostics["status"] = "insufficient_data_after_cleaning"
            fallback_val = df_clean["consumption"].median() if not df_clean.empty else df["consumption"].median()
            return round(fallback_val, 3), diagnostics
        
        # ==========================================
        # ۱. تعیین متغیرهای مستقل کاندید
        # ==========================================
        candidate_vars = []
        
        # اگر کاربر دستی انتخاب کرده باشد، فقط همان ستون
        if manual_x and manual_x != "[تشخیص خودکار]" and manual_x in df_clean.columns:
            candidate_vars = [manual_x]
            diagnostics["notes"].append(f"متغیر دستی: {manual_x}")
        else:
            # تشخیص خودکار: همه ستون‌های عددی به جز مصرف
            for c in df_clean.columns:
                if c not in ["تاریخ", "consumption"] and df_clean[c].nunique() > 3:
                    corr = abs(df_clean["consumption"].corr(df_clean[c])) if df_clean[c].std() > 0 else 0
                    min_corr = 0.3 if disable_filters else 0.82
                    if corr > min_corr:
                        candidate_vars.append(c)
        
        if not candidate_vars:
            diagnostics["notes"].append("متغیر مستقل یافت نشد")
            diagnostics["status"] = "no_suitable_vars"
            fallback_val = df_clean["consumption"].median()
            return round(fallback_val, 3), diagnostics
        
        # ==========================================
        # ۲. بررسی هم‌خطی (VIF) بین متغیرهای کاندید
        # ==========================================
        if len(candidate_vars) > 1:
            X_candidates = df_clean[candidate_vars].dropna()
            if len(X_candidates) > len(candidate_vars):
                X_with_const = sm.add_constant(X_candidates)
                vif_data = pd.DataFrame()
                vif_data["متغیر"] = X_with_const.columns
                vif_data["VIF"] = [variance_inflation_factor(X_with_const.values, i) 
                                   for i in range(X_with_const.shape[1])]
                diagnostics["vif"] = vif_data.set_index("متغیر")["VIF"].to_dict()
                
                high_vif_vars = vif_data[vif_data["VIF"] > 10]["متغیر"].tolist()
                high_vif_vars = [v for v in high_vif_vars if v != "const"]
                if high_vif_vars:
                    diagnostics["notes"].append(f"حذف متغیرهای هم‌خط (VIF>10): {', '.join(high_vif_vars)}")
                    candidate_vars = [v for v in candidate_vars if v not in high_vif_vars]
        
        if not candidate_vars:
            diagnostics["notes"].append("پس از حذف هم‌خطی، متغیری باقی نماند")
            diagnostics["status"] = "no_vars_after_vif"
            fallback_val = df_clean["consumption"].median()
            return round(fallback_val, 3), diagnostics
        
        # ==========================================
        # ۳. اجرای رگرسیون چندمتغیره
        # ==========================================
        X = df_clean[candidate_vars].dropna()
        y = df_clean.loc[X.index, "consumption"]
        
        if len(X) < 5:
            diagnostics["notes"].append("داده کافی برای رگرسیون چندمتغیره")
            diagnostics["status"] = "insufficient_data_for_multi"
            fallback_val = y.median()
            return round(fallback_val, 3), diagnostics
        
        X_const = sm.add_constant(X)
        model = sm.OLS(y, X_const).fit()
        
        # اگر فیلتر غیرفعال باشد، هر مدلی قبول می‌شود
        if disable_filters:
            best_model = model
            best_r2 = model.rsquared
            selected_vars = candidate_vars
            X_final = X_const
            y_final = y
        else:
            if model.f_pvalue < 0.05 and model.rsquared >= 0.67:
                significant_vars = [var for var in candidate_vars 
                                   if var in model.pvalues and model.pvalues[var] < 0.05]
                if significant_vars:
                    X_sig = df_clean[significant_vars].dropna()
                    y_sig = df_clean.loc[X_sig.index, "consumption"]
                    X_sig_const = sm.add_constant(X_sig)
                    model_sig = sm.OLS(y_sig, X_sig_const).fit()
                    best_model = model_sig
                    best_r2 = model_sig.rsquared
                    selected_vars = significant_vars
                    X_final = X_sig_const
                    y_final = y_sig
                else:
                    diagnostics["notes"].append("هیچ متغیری معنی‌دار نشد")
                    diagnostics["status"] = "no_significant_vars"
                    fallback_val = y.median()
                    return round(fallback_val, 3), diagnostics
            else:
                diagnostics["notes"].append(f"مدل معنی‌دار نشد (R²={model.rsquared:.3f}, p={model.f_pvalue:.4f})")
                diagnostics["status"] = "model_not_significant"
                fallback_val = y.median()
                return round(fallback_val, 3), diagnostics
        
        # ==========================================
        # ۴. استخراج ضرایب و معادله
        # ==========================================
        intercept = float(best_model.params.iloc[0])
        coefs = {var: float(best_model.params[var]) for var in selected_vars}
        
        equation_terms = [f"{coefs[var]:.3f}×{var}" for var in selected_vars]
        equation = " + ".join(equation_terms)
        if abs(intercept) > 0.001:
            equation = f"{intercept:.3f} + " + equation
        
        # ==========================================
        # ۵. محاسبه Baseline
        # ==========================================
        if baseline_method == "متغیر (مطابق اکسل - برای هر نقطه)":
            X_vals = df_clean[selected_vars].values
            y_pred = intercept + X_vals @ np.array([coefs[var] for var in selected_vars])
            baseline_val = round(y_pred.mean(), 3)
            diagnostics["baseline_per_point"] = y_pred.tolist()
            diagnostics["notes"].append(f"میانگین EnB نقاط: {baseline_val:.3f}")
        else:
            mean_x = {var: df_clean[var].mean() for var in selected_vars}
            baseline_val = intercept + sum([coefs[var] * mean_x[var] for var in selected_vars])
            baseline_val = round(max(0, baseline_val), 3)
            mean_str = ", ".join([f"{var}={mean_x[var]:.2f}" for var in selected_vars])
            diagnostics["notes"].append(f"EnB در ({mean_str}) → {baseline_val:.3f}")
        
        # ==========================================
        # ۶. محاسبه Residuals (جدید)
        # ==========================================
        # پیش‌بینی‌ها روی داده‌های مورد استفاده در رگرسیون نهایی
        y_pred_all = best_model.predict(X_final)
        residuals_all = y_final - y_pred_all
        
        # ذخیره در دیکشنری
        diagnostics["predictions"] = y_pred_all.tolist()
        diagnostics["residuals"] = residuals_all.tolist()
        diagnostics["actual"] = y_final.tolist()
        diagnostics["observation_indices"] = list(range(1, len(y_final) + 1))  # شماره ردیف‌ها
        
        diagnostics.update({
            "selected_vars": selected_vars,
            "r2": round(best_r2, 3),
            "equation": equation,
            "intercept": round(intercept, 6),
            "coef": coefs,
            "n_points": int(best_model.nobs),
            "status": "regression_success"
        })
        
        return baseline_val, diagnostics
        
    except Exception as e:
        diagnostics.update({"notes": [f"خطا: {str(e)[:100]}"], "status": "error"})
        fallback_val = df_consumption["consumption"].median() if not df_consumption.empty else 0
        return round(fallback_val, 3), diagnostics

# ===============================================
# احراز هویت امن
# - رمزهای عبور دیگر در کد نوشته نمی‌شوند.
# - رمزها (به‌صورت هش‌شده) از فایل .streamlit/secrets.toml خوانده می‌شوند
#   که نباید هرگز در گیت/سورس‌کد commit شود.
# - هش با PBKDF2-HMAC-SHA256 + salt تصادفی انجام می‌شود (مقاوم در برابر brute-force)
#   نه SHA-256 ساده که سریع و قابل کرک با GPU است.
# - برای ساخت هش رمز عبور جدید، اسکریپت generate_password_hash.py را اجرا کنید.
# ===============================================

PBKDF2_ITERATIONS = 200_000

def hash_password(password: str, salt: bytes = None) -> str:
    """هش امن رمز عبور با PBKDF2-HMAC-SHA256 و salt تصادفی.
    خروجی به فرمت 'iterations$salt_hex$hash_hex' ذخیره می‌شود."""
    if salt is None:
        salt = os.urandom(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS)
    return f"{PBKDF2_ITERATIONS}${salt.hex()}${dk.hex()}"

def verify_password(password: str, stored_hash: str) -> bool:
    """مقایسه امن (زمان‌ثابت) رمز عبور وارد شده با هش ذخیره‌شده."""
    try:
        iterations_str, salt_hex, hash_hex = stored_hash.split("$")
        iterations = int(iterations_str)
        salt = bytes.fromhex(salt_hex)
        dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
        return hmac.compare_digest(dk.hex(), hash_hex)
    except Exception:
        return False

@st.cache_resource
def load_users():
    """بارگذاری کاربران و هش رمز عبورشان از st.secrets.
    هرگز رمز عبور یا حتی هش رمز عبور را داخل این فایل کد ننویسید."""
    try:
        users_secrets = st.secrets["users"]
    except Exception:
        st.error(
            "⚠️ فایل تنظیمات کاربران (`.streamlit/secrets.toml`) یافت نشد یا بخش "
            "`[users]` در آن تعریف نشده است.\n\n"
            "لطفاً طبق راهنمای `SECURITY_SETUP.md` این فایل را بسازید و برنامه را دوباره اجرا کنید."
        )
        st.stop()

    users = {}
    for username, info in users_secrets.items():
        users[username] = {
            "password_hash": info["password_hash"],
            "role": info["role"],
        }
    return users

USERS = load_users()
import base64
import os

def get_base64_of_local_file(path):
    try:
        with open(path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')
    except:
        return None
def login():
    # لینک مستقیم لوگو از سرچ (PNG با کیفیت بالا)
    logo_url = "https://brandfetch.com/api/v2/organization/simidco.com/logo.png"

    st.markdown(f"""
    <style>
    /* پس‌زمینه تمام صفحه با لوگوی SIMIDCO از اینترنت */
    .stApp {{
        background: linear-gradient(rgba(0,0,0,0.70), rgba(0,0,0,0.80)),
                    url("{logo_url}") no-repeat center center fixed;
        background-size: contain;  /* لوگو رو بزرگ و متمرکز نگه می‌داره */
        min-height: 100vh;
    }}

    /* مخفی کردن هدر و سایدبار در صفحه لاگین */
    header, footer {{visibility: hidden;}}
    section[data-testid="stSidebar"] {{display: none !important;}}

    /* باکس ورود شیک وسط صفحه */
    .login-container {{
        background: rgba(255, 255, 255, 0.95);
        padding: 45px 55px;
        border-radius: 20px;
        box-shadow: 0 15px 40px rgba(0,0,0,0.5);
        max-width: 460px;
        margin: 100px auto;
        text-align: center;
        border: 4px solid #C74A1B;  /* رنگ نارنجی شرکت */
        backdrop-filter: blur(12px);
    }}

    .login-title {{
        color: #C74A1B;
        font-size: 32px;
        font-weight: bold;
        margin-bottom: 8px;
    }}

    .login-subtitle {{
        color: #1A1A1A;
        font-size: 18px;
        margin-bottom: 35px;
    }}

    /* استایل دکمه ورود */
    div.stButton > button {{
        background-color: #C74A1B;
        color: white;
        border-radius: 10px;
        padding: 12px 30px;
        font-weight: bold;
        border: none;
        width: 100%;
    }}

    div.stButton > button:hover {{
        background-color: #561018;
    }}
    </style>

    <div class="login-container">
        <h2 class="login-title">SIMIDCO</h2>
        <p class="login-subtitle">مجتمع صنعتی و معدنی توسعه فراگیر سناباد</p>
        <h3 style="color:#1A1A1A; margin-bottom:30px;">🔐 ورود به داشبورد پایش انرژی</h3>
    </div>
    """, unsafe_allow_html=True)

    # فرم ورود (وسط صفحه)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<br><br><br>", unsafe_allow_html=True)  # فاصله عمودی

        username = st.text_input("👤 نام کاربری", placeholder="مثال: e.pourkarim")
        password = st.text_input("🔒 رمز عبور", type="password", placeholder="••••••••")

        if st.button("🚀 ورود به سیستم", use_container_width=True, type="primary"):
            if username in USERS:
                if verify_password(password, USERS[username]["password_hash"]):
                    st.session_state["logged_in"] = True
                    st.session_state["username"] = username
                    st.session_state["role"] = USERS[username]["role"]
                    st.success("✅ ورود موفقیت‌آمیز بود!")
                    st.rerun()
                else:
                    st.error("❌ رمز عبور اشتباه است")
            else:
                st.error("❌ کاربر یافت نشد")

    # پایین صفحه – کپی‌رایت
    st.markdown("""
    <div style="text-align:center; color:white; margin-top:100px; opacity:0.8;">
        <p>نسخه ۱.۲ – کمیته انرژی SIMIDCO © ۱۴۰۴</p>
    </div>
    """, unsafe_allow_html=True)

    # ------------------- نمایش لوگو در بالا (اختیاری) -------------------
    st.markdown("""
    <div style="text-align:center; margin-top:50px;">
        <img src="https://i.postimg.cc/3xVZQyYJ/simidco-logo-big.png" width="200">
        <p style="color:white; font-size:14px; margin-top:20px; opacity:0.8;">
            نسخه ۱.۲ – کمیته انرژی SIMIDCO © ۱۴۰۴
        </p>
    </div>
    """, unsafe_allow_html=True)

if "logged_in" not in st.session_state or not st.session_state["logged_in"]:
    login()
    st.stop()

simidco_template = dict(
    layout=go.Layout(
        font=dict(family="Vazir", size=14, color="#28E00F"),
        title=dict(font=dict(size=18, color="#1A1A1A")),

        plot_bgcolor="white",
        paper_bgcolor="white",

        xaxis=dict(
            showgrid=True,
            gridcolor="#E6E6E6",
            zeroline=False,
            linecolor="#1A1A1A",
            ticks="outside"
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="#E6E6E6",
            zeroline=False,
            linecolor="#1A1A1A",
            ticks="outside"
        ),

        legend=dict(
            orientation="h",
            y=-0.2,
            x=0,
            font=dict(size=13)
        ),

        colorway=["#C74A1B", "#1A1A1A", "#561018", "#0080FF", "#00A8E8"]
    )
)

pio.templates["simidco"] = simidco_template
pio.templates.default ="simidco"
# Time Series
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
# Optimization
OPTIMIZATION_AVAILABLE = True  # Assume true after imports
# RTL Support
try:
    import arabic_reshaper
    from bidi.algorithm import get_display
    RTL_AVAILABLE = True
except ImportError:
    RTL_AVAILABLE = False
# Additional libs
try:
    from sklearn.decomposition import PCA
    from statsmodels.formula.api import ols
    from statsmodels.stats.anova import anova_lm
    ADDITIONAL_LIBS_AVAILABLE = True
except ImportError:
    ADDITIONAL_LIBS_AVAILABLE = False
# Openpyxl for Baseline
try:
    from openpyxl import load_workbook
    OPENPYXL_AVAILABLE = True
except ImportError:
    OPENPYXL_AVAILABLE = False
# -------------------- Unit Mapping Configuration (تجهیز‌محور) --------------------
equipment_unit_mapping = {
    "تولید گندله": "تن",
    "مصرف برق": "MWh",
    "خروجی گاز": "m³",
    "هزینه": "تومان",
    "Fe": "%",
    "FeO": "%",
    "Mois.": "%",
    "blaine": "m²/kg",
    "Kiln Main Burner": "m³",
    "production(ton)": "تن",
}
DEFAULT_EQUIPMENT_UNIT = "MWh"
# -------------------- Configuration --------------------
IS_CLOUD = os.environ.get("IS_CLOUD", "false").lower() == "true"
st.set_page_config(
    page_title="داشبورد پایش برق کنسانتره",
    layout="wide",
    initial_sidebar_state="expanded"
)
# -------------------- Font Setup --------------------
def setup_persian_font():
    font_paths = [
        r"I:\data\IranSans.ttf",
        r"I:\data\BNazanin.ttf",
        "IranSans.ttf",
        "BNazanin.ttf"
    ]
    for path in font_paths:
        if os.path.exists(path):
            try:
                font_name = "PersianFont"
                pdfmetrics.registerFont(TTFont(font_name, path))
                return font_name, True
            except Exception as e:
                st.sidebar.warning(f"خطا در لود فونت {path}: {e}")
    return "Helvetica", False
FONT_NAME, USE_PERSIAN = setup_persian_font()
# -------------------- Translation Dictionary --------------------
translations = {
    "تجهیز": "Equipment",
    "میانگین مصرف": "Avg Consumption",
    "تاریخ نمایش": "Display Date",
    "مصرف (MWh)": "Consumption (MWh)",
    "ماه شمسی": "Jalali Month",
    "درصد تغییر": "Percent Change",
    "تاریخ": "Date",
    "مجموع": "Total",
    "میانگین": "Average",
    "بیشترین": "Max",
    "هزینه (تومان)": "Cost (Toman)",
    "دوره": "Period",
    "Variable": "Variable",
    "R²": "R²",
    "p-value": "p-value",
    "Impactful": "Impactful",
    "پیش‌بینی مصرف تجهیزات": "Equipment Consumption Forecast",
    "تحلیل روند تغییرات": "Trend Change Analysis",
    "پیش‌بینی": "Forecast",
    "جدول خطاها": "Error Table",
    "تحلیل دیتا": "Data Analysis",
    "تشخیص ناهنجاری‌ها": "Anomaly Detection",
    "گزارش زیست‌محیطی": "Environmental Report",
    "میانگین مصرف تجهیزات": "Equipment Average Consumption",
    "روند مصرف": "Consumption Trend",
    "نمودار مصرف ماهیانه": "Monthly Consumption Chart",
    "Heatmap مصرف تجهیزات": "Equipment Consumption Heatmap",
    "خروجی داده‌ها": "Data Export",
    "KPI پیشرفته": "Advanced KPI",
    "تغییرات درصدی": "Percent Changes",
    "ML پیش‌بینی": "ML Forecast",
    "تحلیل دیتا": "Data Analysis",
    "ناهنجاری‌ها": "Anomalies",
    "زیست‌محیطی": "Environmental",
    "استانداردها": "Standards",
    "هزینه": "Cost",
    "داشبورد زنده": "Live Dashboard",
    "گزارش سفارشی": "Custom Report",
    "شبیه‌سازی": "Simulation",
    "بهینه‌سازی": "Optimization",
    "برق مصرفی گندله": "Pellet electricity consumption",
    "گاز مصرفی گندله": "Pellet gas consumption",
    "برق مصرفی کنسانتره": "Electricity consumption of the concentrate",
    "مقدار برق مصرفی سنگ شکن" : "Amount of electricity consumed by a stone crusher",
    "مقدار برق مصرفی فیلتراسیون" : "Filtration power consumption",
    "گزارش میزان مصرف گازوئیل" : "Diesel consumption report",
}
def load_simidco_theme():
    st.markdown("""
    <style>

    /* --------------------------------------- */
    /* فونت فارسی (IRANSans یا Vazir)          */
    /* --------------------------------------- */
    @import url('https://cdn.jsdelivr.net/gh/rastikerdar/vazir-font@v30.1.0/dist/font-face.css');
    * {
        font-family: Vazir !important;
    }

    /* --------------------------------------- */
    /* رنگ سازمانی SIMIDCO                     */
    /* --------------------------------------- */
    :root {
        --sim-orange: #C74A1B;
        --sim-black:  #1A1A1A;
        --sim-maroon: #561018;
        --sim-gray:   #E6E6E6;
    }

    /* --------------------------------------- */
    /* استایل کارت KPI                         */
    /* --------------------------------------- */
    div.stMetric {
        background-color: white;
        padding: 10px 15px;
        border-radius: 10px;
        border-left: 6px solid var(--sim-orange);
        box-shadow: 0 0 8px rgba(0,0,0,0.08);
    }

    /* --------------------------------------- */
    /* دکمه‌ها                                   */
    /* --------------------------------------- */
    div.stButton > button {
        background-color: var(--sim-black);
        color: white;
        border-radius: 8px;
        padding: 8px 25px;
        border: none;
    }

    div.stButton > button:hover {
        background-color: var(--sim-maroon);
        color: white;
    }

    /* --------------------------------------- */
    /* تب‌ها                                    */
    /* --------------------------------------- */
    .stTabs [data-baseweb="tab"] {
        color: var(--sim-black);
        font-weight: 600;
    }

    .stTabs [aria-selected="true"] {
        color: var(--sim-orange);
        border-bottom: 3px solid var(--sim-orange);
    }

    /* --------------------------------------- */
    /* Sidebar                                  */
    /* --------------------------------------- */
    section[data-testid="stSidebar"] {
        background-color: #f8f8f8;
        border-right: 2px solid var(--sim-gray);
    }

    /* --------------------------------------- */
    /* Header                                    */
    /* --------------------------------------- */
    header[data-testid="stHeader"] {
        background-color: white !important;
        border-bottom: 2px solid var(--sim-gray);
    }

    </style>
    """, unsafe_allow_html=True)
    load_simidco_theme()
# -------------------- Helper Functions --------------------
def reshape_rtl(text):
    if RTL_AVAILABLE and USE_PERSIAN and isinstance(text, str):
        try:
            return get_display(arabic_reshaper.reshape(text))
        except:
            return text
    return text

# نام مستعار rtl: در چند تب برای متن‌های جدول PDF از rtl() استفاده شده بود
# بدون این‌که جایی تعریف شده باشد (باعث NameError هنگام دانلود PDF می‌شد).
rtl = reshape_rtl
def generate_pdf(title, elements, buffer):
    try:
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        styles = getSampleStyleSheet()
        if USE_PERSIAN:
            title_style = ParagraphStyle(
                'TitlePersian',
                parent=styles['Title'],
                fontName=FONT_NAME,
                fontSize=18,
                alignment=1
            )
        else:
            title_style = styles['Title']
            title = translations.get(title, title)
        title_para = Paragraph(reshape_rtl(title), title_style)
        all_elements = [title_para, Spacer(1, 12)] + elements
        doc.build(all_elements)
    except Exception as e:
        st.error(f"خطا در تولید PDF: {e}")
def get_unit_for_column(df, column_name, custom_units=None):
    if custom_units and column_name in custom_units:
        return custom_units[column_name]
    if column_name in equipment_unit_mapping:
        return equipment_unit_mapping[column_name]
    return DEFAULT_EQUIPMENT_UNIT
def safe_jalali_format(date):
    if pd.isna(date) or date is None:
        return ""
    try:
        return JalaliDate(date).strftime('%Y/%m/%d')
    except:
        return str(date)
def load_excel(file):
    try:
        dfs = []
        xls = pd.ExcelFile(file)
        for sheet in xls.sheet_names:
            df_sheet = pd.read_excel(file, sheet_name=sheet, header=None)
            df_sheet = df_sheet.dropna(axis=1, how="all")
            if df_sheet.empty:
                continue
            header_row = 2
            if len(df_sheet) < header_row + 1:
                continue
            raw_headers = df_sheet.iloc[header_row].fillna('')
            seen, unique_headers = {}, []
            for col in raw_headers:
                col_str = str(col)
                if col_str not in seen:
                    seen[col_str] = 0
                    unique_headers.append(col_str)
                else:
                    seen[col_str] += 1
                    unique_headers.append(f"{col_str}*{seen[col_str]}")
            df_data = df_sheet[header_row+1:].copy()
            df_data.columns = unique_headers
            if df_data.columns[0] != "تاریخ":
                df_data = df_data.rename(columns={df_data.columns[0]: "تاریخ"})
            df_data["تاریخ"] = pd.to_datetime(df_data["تاریخ"], errors="coerce")
            df_data = df_data.dropna(subset=["تاریخ"])
            if df_data.empty:
                continue
            df_data["تاریخ شمسی"] = df_data["تاریخ"].apply(safe_jalali_format)
            for col in df_data.columns:
                if col not in ["تاریخ", "تاریخ شمسی"]:
                    df_data[col] = pd.to_numeric(df_data[col], errors="coerce")
            df_data["کارخانه"] = sheet if sheet else "نامشخص"
            dfs.append(df_data)
        if not dfs:
            return pd.DataFrame()
        return pd.concat(dfs, ignore_index=True)
    except Exception as e:
        st.error(f"خطا در بارگذاری فایل اکسل: {e}")
        return pd.DataFrame()
def monte_carlo_simulation(base_values, scenarios, n_simulations=1000, change_factor=0.1):
    if len(base_values) == 0 or len(scenarios) == 0:
        return pd.DataFrame()
    results = []
    for scenario in scenarios:
        sim_data = []
        for val in base_values:
            sim_vals = val * (1 + scenario + np.random.normal(0, change_factor, n_simulations))
            sim_data.append(sim_vals)
        sim_means = np.mean(sim_data, axis=0)
        results.append(sim_means)
    sim_df = pd.DataFrame(
        np.array(results).T,
        columns=[f"Scenario*{i+1}" for i in range(len(scenarios))]
    )
    return sim_df
# -------------------- Header --------------------
st.markdown("""
    <div style="background-color:#C74A1B;padding:20px;border-radius:10px;text-align:center;">
        <h1 style="color:white;font-family:sans-serif;">مجتمع صنعتی و معدنی توسعه فراگیر سناباد</h1>

""", unsafe_allow_html=True)
# -------------------- Sidebar: Logo Upload --------------------
st.sidebar.subheader("🏷️ بارگذاری لوگو شرکت")
uploaded_logo = st.sidebar.file_uploader(
    "آپلود لوگو (PNG/JPG)",
    type=["png", "jpg", "jpeg"],
    key="logo_uploader_main"
)
if uploaded_logo:
    try:
        st.sidebar.image(uploaded_logo, width=150)
    except Exception as e:
        st.sidebar.error(f"خطا در بارگذاری لوگو: {e}")
else:
    logo_path = r"I:\data\logo.png"
    if os.path.exists(logo_path):
        st.sidebar.image(logo_path, width=150)
# -------------------- File Upload --------------------
uploaded_file = st.file_uploader(
    "📂 لطفاً فایل اکسل کنسانتره را بارگذاری کنید",
    type=["xlsx"],
    key="excel_uploader_main"
)
if uploaded_file is None:
    default_path = r"I:\data\کنسانتره_پایش.xlsx"
    if os.path.exists(default_path):
        uploaded_file = default_path
        st.info("📌 از فایل پیش‌فرض استفاده شد.")
    else:
        st.warning("⚠️ فایل اکسل بارگذاری نشده است.")
        st.stop()
# -------------------- Load Data --------------------
with st.spinner("در حال بارگذاری داده‌ها..."):
    df = load_excel(uploaded_file)
if df.empty:
    st.error("⚠️ هیچ داده‌ای از فایل اکسل بارگذاری نشد.")
    st.stop()
# -------------------- Sidebar Filters --------------------
st.sidebar.header("🏭 انتخاب کارخانه")
factories = df["کارخانه"].unique().tolist()
if st.sidebar.button("انتخاب همه کارخانه‌ها", key="select_all_factories_btn"):
    selected_factories = factories
else:
    selected_factories = st.sidebar.multiselect(
        "انتخاب کارخانه",
        factories,
        default=factories,
        key="multiselect_factories_main"
    )
if not selected_factories:
    st.warning("⚠️ لطفاً حداقل یک کارخانه انتخاب کنید.")
    st.stop()
filtered_df = df[df["کارخانه"].isin(selected_factories)].copy()
# -------------------- Equipment Columns --------------------
numeric_columns = filtered_df.select_dtypes(include=['number']).columns.tolist()
equipment_columns = [col for col in numeric_columns if col not in ['کارخانه']]
# -------------------- Date Filter --------------------
st.sidebar.header("🎯 بازه زمانی")
min_date = filtered_df["تاریخ"].min().date()
max_date = filtered_df["تاریخ"].max().date()
date_range = st.sidebar.date_input(
    "بازه زمانی",
    [min_date, max_date],
    key="date_range_main"
)
if len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date = end_date = date_range[0]

# 👈 اصلاح: پرانتز خارجی برای جلوگیری از خطای line continuation
mask = ((filtered_df["تاریخ"] >= pd.to_datetime(start_date)) & 
        (filtered_df["تاریخ"] <= pd.to_datetime(end_date)))
filtered_df = filtered_df.loc[mask].copy()
if filtered_df.empty:
    st.warning("⚠️ داده‌ای برای بازه انتخاب‌شده وجود ندارد.")
    st.stop()
st.sidebar.markdown(f"**تاریخ شمسی شروع:** {safe_jalali_format(start_date)}")
st.sidebar.markdown(f"**تاریخ شمسی پایان:** {safe_jalali_format(end_date)}")
# -------------------- Equipment Selection --------------------
st.sidebar.header("🔌 انتخاب تجهیزات")
selected_equipment = st.sidebar.multiselect(
    "انتخاب تجهیز(ها):",
    options=equipment_columns,
    default=equipment_columns[:3] if len(equipment_columns) >= 3 else equipment_columns,
    key="equipment_multiselect_main"
)
if not selected_equipment:
    st.warning("⚠️ لطفاً حداقل یک تجهیز انتخاب کنید.")
    st.stop()
# Custom units expander
if 'custom_units' not in st.session_state:
    st.session_state.custom_units = {}
with st.sidebar.expander("📏 ویرایش واحد تجهیزات", expanded=False):
    for eq in selected_equipment:
        default_unit = get_unit_for_column(filtered_df, eq, st.session_state.custom_units)
        new_unit = st.text_input(
            f"واحد {eq}:",
            value=default_unit,
            key=f"unit_input_{eq}"
        )
        if new_unit:
            st.session_state.custom_units[eq] = new_unit
        else:
            if eq in st.session_state.custom_units:
                del st.session_state.custom_units[eq]
selected_equipment_units = [get_unit_for_column(filtered_df, eq, st.session_state.custom_units) for eq in selected_equipment]
unique_units = list(set(selected_equipment_units))
if len(unique_units) > 1:
    st.sidebar.warning(f"⚠️ واحدهای متفاوت در تجهیزات انتخاب‌شده: {', '.join(unique_units)}. تحلیل ممکنه نیاز به تنظیم داشته باشه.")
# -------------------- Tabs --------------------
tabs = st.tabs([
    "📊 KPI & مقایسه",
    "📈 روند مصرف",
    "📆 ماهانه",
    "🔥 Heatmap",
    "📝 جدول & خروجی",
    "🔮 پیش‌بینی مصرف تجهیزات",
    "✨ KPI پیشرفته",
    "📈 تغییرات درصدی",
    "🤖 ML پیش‌بینی",
    "🔬 تحلیل دیتا",
    "🚨 ناهنجاری‌ها",
    "🌍 زیست‌محیطی",
    "🏭 استانداردها",
    "💰 تحلیل هزینه و بودجه",
    "📱 داشبورد زنده",
    "📱 گزارش سفارشی",
    "🎲 شبیه‌سازی سناریوها",
    "⚙️ بهینه‌سازی",
    "🕵️ شناسایی و تحلیل وابستگی‌ها",
    "📊 خط مبنا (Baseline)",
    "📋 گزارش EnMS برای ISO 50001 (PDCA)",
    "📋 فرمول‌های محاسباتی برای ممیزی انرژی",
    "برنامه عملیاتی",
])
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots   # <--- این خط را اضافه کن
import datetime
# ... بقیه importها
with tabs[0]:
    st.subheader("مقایسه میانگین مصرف تجهیزات")
    
    if not selected_equipment:
        st.info("لطفاً از سایدبار حداقل یک تجهیز انتخاب کنید.")
    elif filtered_df.empty or filtered_df[selected_equipment].dropna(how="all").empty:
        st.warning("⚠️ در بازه تاریخ انتخاب‌شده هیچ داده‌ای برای تجهیزات انتخابی وجود ندارد. لطفاً بازه تاریخ یا تجهیزات را تغییر دهید.")
    else:
        # ------------------------------------------------------------------
        # محاسبه میانگین و مجموع مصرف
        # ------------------------------------------------------------------
        mean_values = filtered_df[selected_equipment].mean(numeric_only=True).reset_index()
        mean_values.columns = ["تجهیز", "میانگین مصرف"]
        mean_values["میانگین مصرف"] = mean_values["میانگین مصرف"].round(2)

        sum_values = filtered_df[selected_equipment].sum(numeric_only=True).reset_index()
        sum_values.columns = ["تجهیز", "مجموع مصرف"]
        sum_values["مجموع مصرف"] = sum_values["مجموع مصرف"].round(2)

        # واحد برای هر تجهیز
        mean_values["واحد"] = [get_unit_for_column(filtered_df, eq, st.session_state.custom_units) for eq in mean_values["تجهیز"]]

        # ادغام میانگین و مجموع
        comparison_df = pd.merge(mean_values[["تجهیز", "میانگین مصرف", "واحد"]], sum_values, on="تجهیز")

        # محاسبه مجموع کل
        total_mean_all = comparison_df["میانگین مصرف"].sum()
        total_sum_all = comparison_df["مجموع مصرف"].sum()

        # واحد غالب
        unit_counts = comparison_df["واحد"].value_counts()
        dominant_unit = unit_counts.index[0] if len(unit_counts) > 0 else ""

        # نمایش با واحد
        comparison_display = comparison_df.copy()
        comparison_display["میانگین مصرف (با واحد)"] = comparison_display.apply(
            lambda row: f"{row['میانگین مصرف']:,} {row['واحد']}", axis=1
        )
        comparison_display["مجموع مصرف (با واحد)"] = comparison_display.apply(
            lambda row: f"{row['مجموع مصرف']:,} {row['واحد']}", axis=1
        )

        # ردیف مجموع کل
        total_row = {
            "تجهیز": "🟰 **مجموع کل تمام تجهیزات**",
            "میانگین مصرف": total_mean_all,
            "مجموع مصرف": total_sum_all,
            "واحد": dominant_unit,
            "میانگین مصرف (با واحد)": f"**{total_mean_all:,.2f} {dominant_unit}**",
            "مجموع مصرف (با واحد)": f"**{total_sum_all:,.2f} {dominant_unit}**"
        }

        # ------------------------------------------------------------------
        # نمودار میله‌ای میانگین مصرف
        # ------------------------------------------------------------------
        st.markdown("### 📊 نمودار میانگین مصرف")
        chart_height = max(400, len(selected_equipment) * 60)
        fig_bar_mean = go.Figure(data=[
            go.Bar(
                x=comparison_df["تجهیز"],
                y=comparison_df["میانگین مصرف"],
                text=comparison_display["میانگین مصرف (با واحد)"],
                textposition="outside",
                marker_color="#C74A1B",
                hovertemplate="<b>%{x}</b><br>میانگین: %{text}<extra></extra>",
            )
        ])
        fig_bar_mean.add_hline(
            y=total_mean_all / len(selected_equipment),
            line_dash="dash", line_color="red",
            annotation_text=f"میانگین کل: {total_mean_all / len(selected_equipment):,.1f}",
            annotation_position="top right"
        )
        fig_bar_mean.update_layout(
            title="میانگین مصرف تجهیزات در بازه انتخابی",
            xaxis=dict(title="تجهیز", tickangle=-45 if len(selected_equipment) > 3 else 0),
            yaxis=dict(title="میانگین مصرف", range=[0, comparison_df["میانگین مصرف"].max() * 1.15]),
            plot_bgcolor='white', paper_bgcolor='white', height=chart_height, margin=dict(t=100, b=100)
        )
        st.plotly_chart(fig_bar_mean, use_container_width=True)

        # ------------------------------------------------------------------
        # نمودار میله‌ای مجموع مصرف
        # ------------------------------------------------------------------
        st.markdown("### 📊 نمودار مجموع مصرف")
        fig_bar_sum = go.Figure(data=[
            go.Bar(
                x=comparison_df["تجهیز"],
                y=comparison_df["مجموع مصرف"],
                text=comparison_display["مجموع مصرف (با واحد)"],
                textposition="outside",
                marker_color="#2E86AB",
                hovertemplate="<b>%{x}</b><br>مجموع: %{text}<extra></extra>",
            )
        ])
        fig_bar_sum.add_hline(
            y=total_sum_all, line_dash="dot", line_color="green",
            annotation_text=f"مجموع کل: {total_sum_all:,.0f}", annotation_position="top right"
        )
        fig_bar_sum.update_layout(
            title="مجموع مصرف تجهیزات در بازه انتخابی",
            xaxis=dict(title="تجهیز", tickangle=-45 if len(selected_equipment) > 3 else 0),
            yaxis=dict(title="مجموع مصرف", range=[0, comparison_df["مجموع مصرف"].max() * 1.15]),
            plot_bgcolor='white', paper_bgcolor='white', height=chart_height, margin=dict(t=100, b=100)
        )
        st.plotly_chart(fig_bar_sum, use_container_width=True)

        # ------------------------------------------------------------------
        # نمودار ترکیبی
        # ------------------------------------------------------------------
        st.markdown("### 📊 نمودار ترکیبی (میانگین و مجموع)")
        fig_combo = make_subplots(specs=[[{"secondary_y": True}]])
        fig_combo.add_trace(go.Bar(x=comparison_df["تجهیز"], y=comparison_df["میانگین مصرف"],
                                   name="میانگین مصرف", marker_color="#C74A1B"), secondary_y=False)
        fig_combo.add_trace(go.Scatter(x=comparison_df["تجهیز"], y=comparison_df["مجموع مصرف"],
                                       name="مجموع مصرف", mode="lines+markers", line=dict(color="#2E86AB", width=3),
                                       marker=dict(size=10, symbol="diamond")), secondary_y=True)
        fig_combo.update_layout(title="مقایسه میانگین و مجموع مصرف تجهیزات",
                                xaxis=dict(title="تجهیز", tickangle=-45 if len(selected_equipment) > 3 else 0),
                                plot_bgcolor='white', paper_bgcolor='white', height=500,
                                legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.8)"), hovermode="x unified")
        fig_combo.update_yaxes(title_text="میانگین مصرف", title_font=dict(color="#C74A1B"), tickfont=dict(color="#C74A1B"), secondary_y=False)
        fig_combo.update_yaxes(title_text="مجموع مصرف", title_font=dict(color="#2E86AB"), tickfont=dict(color="#2E86AB"), secondary_y=True)
        st.plotly_chart(fig_combo, use_container_width=True)

        # ------------------------------------------------------------------
        # جداول مقایسه
        # ------------------------------------------------------------------
        st.markdown("### 📋 جداول مقایسه")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("جدول عددی (با گرادیان)")
            styled_df = comparison_df[["تجهیز", "میانگین مصرف", "مجموع مصرف"]].copy()
            total_df = pd.DataFrame([{"تجهیز": "🟰 مجموع کل", "میانگین مصرف": total_mean_all, "مجموع مصرف": total_sum_all}])
            styled_df = pd.concat([styled_df, total_df], ignore_index=True)
            styled_table = styled_df.style.format({"میانگین مصرف": "{:,.2f}", "مجموع مصرف": "{:,.2f}"}) \
                .background_gradient(cmap="Oranges", subset="میانگین مصرف") \
                .background_gradient(cmap="Blues", subset="مجموع مصرف") \
                .apply(lambda x: ['background: #e8f5e8; font-weight: bold' if x.name == len(styled_df)-1 else '' for _ in x], axis=1)
            st.dataframe(styled_table, use_container_width=True, height=400)

        with col2:
            st.subheader("نمایش با واحد")
            display_df = comparison_display[["تجهیز", "میانگین مصرف (با واحد)", "مجموع مصرف (با واحد)"]].copy()
            total_display = pd.DataFrame([total_row])
            display_df = pd.concat([display_df[["تجهیز", "میانگین مصرف (با واحد)", "مجموع مصرف (با واحد)"]], total_display[["تجهیز", "میانگین مصرف (با واحد)", "مجموع مصرف (با واحد)"]]], ignore_index=True)
            def highlight(row):
                return ['background-color: #e8f5e8; font-weight: bold'] * len(row) if "مجموع کل" in row["تجهیز"] else [''] * len(row)
            st.dataframe(display_df.style.apply(highlight, axis=1), use_container_width=True, height=400)

        # ------------------------------------------------------------------
        # خلاصه آماری
        # ------------------------------------------------------------------
        st.markdown("### 📊 خلاصه آماری")
        col1, col2, col3, col4 = st.columns(4)
        avg_all = comparison_df["میانگین مصرف"].mean()
        max_eq = comparison_df.loc[comparison_df["مجموع مصرف"].idxmax()]
        contrib = (max_eq["مجموع مصرف"] / total_sum_all * 100) if total_sum_all > 0 else 0

        with col1: st.metric("تعداد تجهیزات", len(selected_equipment))
        with col2: st.metric("مجموع مصرف کل", f"{total_sum_all:,.0f}", dominant_unit)
        with col3: st.metric("میانگین کل تجهیزات", f"{avg_all:,.1f}", dominant_unit)
        with col4: st.metric("بیشترین مصرف", max_eq["تجهیز"], f"{contrib:.1f}% از کل")

        # ------------------------------------------------------------------
        # تحلیل توزیع مصرف (پارتو) – این بخش قبلاً حذف شده بود، حالا برگشت!
        # ------------------------------------------------------------------
        st.markdown("### 📈 تحلیل توزیع مصرف")
        if total_sum_all > 0:
            comparison_df["سهم از کل %"] = (comparison_df["مجموع مصرف"] / total_sum_all * 100).round(1)
        else:
            comparison_df["سهم از کل %"] = 0.0
        comparison_df["سهم تجمعی %"] = comparison_df["سهم از کل %"].cumsum()

        col_dist1, col_dist2 = st.columns(2)

        with col_dist1:
            fig_pareto = make_subplots(specs=[[{"secondary_y": True}]])
            fig_pareto.add_trace(go.Bar(x=comparison_df["تجهیز"], y=comparison_df["سهم از کل %"],
                                        name="سهم هر تجهیز", marker_color="#FF6B6B"), secondary_y=False)
            fig_pareto.add_trace(go.Scatter(x=comparison_df["تجهیز"], y=comparison_df["سهم تجمعی %"],
                                            name="سهم تجمعی", mode="lines+markers", line=dict(color="#4ECDC4", width=3)), secondary_y=True)
            fig_pareto.update_layout(title="تحلیل پارتو: سهم هر تجهیز از مصرف کل", xaxis=dict(title="تجهیز", tickangle=-45),
                                     height=450, legend=dict(x=0.02, y=0.98))
            fig_pareto.update_yaxes(title_text="سهم هر تجهیز (%)", range=[0, 100], secondary_y=False)
            fig_pareto.update_yaxes(title_text="سهم تجمعی (%)", range=[0, 100], secondary_y=True)
            st.plotly_chart(fig_pareto, use_container_width=True)

        with col_dist2:
            dist_df = comparison_df[["تجهیز", "مجموع مصرف", "سهم از کل %", "سهم تجمعی %"]].sort_values("سهم از کل %", ascending=False).copy()
            total_dist = pd.DataFrame([{"تجهیز": "**مجموع کل**", "مجموع مصرف": total_sum_all, "سهم از کل %": 100.0, "سهم تجمعی %": 100.0}])
            dist_df = pd.concat([dist_df, total_dist], ignore_index=True)
            def highlight_total(row):
                return ['background-color: #e8f5e8; font-weight: bold'] * len(row) if "مجموع کل" in str(row["تجهیز"]) else [''] * len(row)
            st.dataframe(dist_df.style.format({"مجموع مصرف": "{:,.0f}", "سهم از کل %": "{:.1f}%", "سهم تجمعی %": "{:.1f}%"})
                         .apply(highlight_total, axis=1), use_container_width=True, height=450)
            st.caption("📊 اصل پارتو: معمولاً ۲۰٪ تجهیزات مسئول ۸۰٪ مصرف کل هستند")

        # ------------------------------------------------------------------
        # دکمه دانلود PDF (همان قبلی)
        # ------------------------------------------------------------------
        st.markdown("---")
        if st.button("📥 دانلود گزارش PDF", key="download_tab1_pdf", type="primary"):
            pdf_buffer = io.BytesIO()
            pdf_table_data = [["تجهیز", "میانگین مصرف", "مجموع مصرف", "واحد"]]
            for _, r in comparison_df.iterrows():
                pdf_table_data.append([
                    str(r["تجهیز"]), f"{r['میانگین مصرف']:,.2f}", f"{r['مجموع مصرف']:,.2f}", str(r["واحد"])
                ])
            pdf_table_data.append(["مجموع کل", f"{total_mean_all:,.2f}", f"{total_sum_all:,.2f}", dominant_unit])

            pdf_table = Table(pdf_table_data, hAlign="CENTER")
            pdf_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#C74A1B")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("BACKGROUND", (0, -1), (-1, -1), colors.HexColor("#e8f5e8")),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ]))

            generate_pdf("مقایسه میانگین مصرف تجهیزات", [pdf_table], pdf_buffer)
            st.download_button(
                "⬇️ ذخیره فایل PDF",
                data=pdf_buffer.getvalue(),
                file_name="gozaresh_moghayese_masraf.pdf",
                mime="application/pdf",
                key="download_tab1_pdf_actual",
            )

# ===============================================
# Tab 2: روند مصرف تجهیزات — کاملاً شمسی + بدون خطا
# ===============================================
with tabs[1]:
    st.subheader("روند مصرف تجهیزات در طول زمان")
    selected_multi = st.multiselect(
        "انتخاب تجهیز(ها) برای نمایش روند:",
        options=selected_equipment,
        default=selected_equipment[:2] if len(selected_equipment) >= 2 else selected_equipment,
        key="trend_equipment_select"
    )
    if not selected_multi:
        st.info("لطفاً حداقل یک تجهیز انتخاب کنید.")
    else:
        time_granularity = st.radio(
            "بازه زمانی تجمیع:",
            options=["روزانه", "ماهانه", "سالیانه"],
            index=1,
            horizontal=True,
            key="trend_granularity"
        )
        # آماده‌سازی داده با تاریخ کاملاً شمسی
        df_plot = filtered_df.copy()
        if time_granularity == "روزانه":
            df_plot["تاریخ نمایش"] = df_plot["تاریخ"].apply(safe_jalali_format) # مثل ۱۴۰۴/۰۳/۰۵
            group_col = "تاریخ نمایش"
            date_format = "روزانه"
        elif time_granularity == "ماهانه":
            # تبدیل به ماه شمسی (مثل ۱۴۰۴/۰۳)
            df_plot["تاریخ نمایش"] = df_plot["تاریخ"].apply(
                lambda x: safe_jalali_format(x)[:7] if pd.notnull(x) else None
            )
            group_col = "تاریخ نمایش"
            date_format = "ماهانه"
        else: # سالیانه
            # فقط سال شمسی (مثل ۱۴۰۴)
            df_plot["تاریخ نمایش"] = df_plot["تاریخ"].apply(
                lambda x: str(JalaliDate(x).year) if pd.notnull(x) else None
            )
            group_col = "تاریخ نمایش"
            date_format = "سالیانه"
        # گروه‌بندی و جمع (برای هر سه بازه، تا ردیف‌های تکراری یک دوره با هم جمع شوند)
        df_plot = df_plot.groupby(group_col, as_index=False)[selected_multi].sum()
        if df_plot.empty:
            st.warning("⚠️ برای بازه و تجهیزات انتخاب‌شده داده‌ای برای نمایش روند وجود ندارد.")
        else:
            # رسم نمودار
            fig_line = go.Figure()
            plotly_palette = ["#C74A1B", "#1A1A1A", "#0080FF", "#00A8E8", "#2E7D32", "#9C27B0"]
            for i, col in enumerate(selected_multi):
                unit = get_unit_for_column(filtered_df, col, st.session_state.custom_units)
                fig_line.add_trace(go.Scatter(
                    x=df_plot["تاریخ نمایش"],
                    y=df_plot[col],
                    mode="lines+markers",
                    name=f"{col} ({unit})",
                    line=dict(color=plotly_palette[i % len(plotly_palette)], width=3),
                    marker=dict(size=9),
                    hovertemplate=f"<b>{col}</b><br>تاریخ: %{{x}}<br>مصرف: %{{y:,.0f}} {unit}<extra></extra>"
                ))
            fig_line.update_layout(
                title=f"روند مصرف تجهیزات — {date_format} (تقویم شمسی)<br>"
                      f"<sub>بازه: {safe_jalali_format(start_date)} تا {safe_jalali_format(end_date)}</sub>",
                xaxis_title="تاریخ شمسی",
                yaxis_title="مصرف",
                hovermode="x unified",
                template="plotly_white",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                height=600,
                margin=dict(t=130)
            )
            st.plotly_chart(fig_line, use_container_width=True)
            # خلاصه آماری
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("تعداد دوره", len(df_plot))
            with col2:
                if len(df_plot) > 1:
                    total_change = ((df_plot[selected_multi].iloc[-1] - df_plot[selected_multi].iloc[0]) /
                                   df_plot[selected_multi].iloc[0].replace(0, np.nan) * 100)
                    avg_change = total_change.mean()
                    st.metric("میانگین تغییر کل دوره", f"{avg_change:+.1f}%")
                else:
                    st.metric("تغییر", "داده ناکافی")
            with col3:
                st.metric("آخرین دوره", df_plot["تاریخ نمایش"].iloc[-1])
            # دانلود PDF حرفه‌ای
            if st.button("دانلود گزارش روند مصرف (PDF)", key="download_trend_pdf"):
                buffer = io.BytesIO()
                elements = []
                header = f"""
                <b>گزارش روند مصرف تجهیزات</b><br/>
                نوع نمایش: {date_format} (تقویم شمسی)<br/>
                بازه: {safe_jalali_format(start_date)} تا {safe_jalali_format(end_date)} | تعداد دوره: {len(df_plot)}<br/>
                تاریخ گزارش: {safe_jalali_format(pd.Timestamp.today())}
                """
                elements.append(Paragraph(rtl(header), ParagraphStyle('Normal', fontName=FONT_NAME, fontSize=12, alignment=1, spaceAfter=25)))
                # جدول خلاصه
                summary_data = [["تجهیز", "آخرین مقدار", "واحد"]]
                for col in selected_multi:
                    unit = get_unit_for_column(filtered_df, col, st.session_state.custom_units)
                    last_val = df_plot[col].iloc[-1]
                    summary_data.append([rtl(col), rtl(f"{last_val:,.0f}"), rtl(unit)])
                table = Table(summary_data, colWidths=[200, 150, 100])
                table.setStyle(TableStyle([
                    ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#C74A1B")),
                    ('TEXTCOLOR', (0,0), (-1,0), colors.white),
                    ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                    ('FONTNAME', (0,0), (-1,-1), FONT_NAME),
                    ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
                ]))
                elements.append(table)
                elements.append(Spacer(1, 20))
                # تصویر نمودار
                try:
                    img_buf = io.BytesIO()
                    fig_line.write_image(img_buf, format="png", engine="kaleido", width=1200, height=650, scale=3)
                    img_buf.seek(0)
                    elements.append(Image(img_buf, width=550, height=320))
                except Exception as e:
                    st.warning(f"نمودار در PDF اضافه نشد: {e}")
                generate_pdf("گزارش روند مصرف تجهیزات", elements, buffer)
                buffer.seek(0)
                st.download_button(
                    "دانلود گزارش روند مصرف (PDF)",
                    buffer.getvalue(),
                    f"روند_مصرف_{safe_jalali_format(pd.Timestamp.today())}.pdf",
                    "application/pdf",
                    key="trend_pdf_final"
                )
    # ===============================================
# Tab 3: تحلیل مصرف زمانی (ماهانه، فصلی، سالانه)
# ===============================================
with tabs[2]:
    st.subheader("تحلیل مصرف تجهیزات در مقیاس زمانی (ماهانه / فصلی / سالانه)")
    if not selected_equipment:
        st.info("لطفاً از سایدبار حداقل یک تجهیز انتخاب کنید.")
    else:
        # انتخاب تجهیز
        monthly_column = st.selectbox(
            "انتخاب تجهیز برای تحلیل:",
            options=selected_equipment,
            index=0,
            key="monthly_equipment_select"
        )
        unit = get_unit_for_column(filtered_df, monthly_column, st.session_state.custom_units)
        # استخراج سال‌های شمسی (روی یک کپی محلی، تا دیتافریم مشترک بین تب‌ها دستکاری نشود)
        def get_jalali_year(date):
            try:
                return JalaliDate(date).year if pd.notnull(date) else None
            except:
                return None
        df_tab3 = filtered_df.copy()
        df_tab3["سال_شمسی"] = df_tab3["تاریخ"].apply(get_jalali_year)
        available_years = sorted([y for y in df_tab3["سال_شمسی"].dropna().unique() if y])
        if not available_years:
            st.error("هیچ سال شمسی معتبری در داده‌ها یافت نشد.")
        else:
            # هشدار سال جاری ناقص
            current_year = get_jalali_year(pd.Timestamp.now())
            if current_year in available_years:
                latest = df_tab3[df_tab3["سال_شمسی"] == current_year]["تاریخ"].max()
                if pd.notnull(latest):
                    latest_month = int(safe_jalali_format(latest).split("/")[1])
                    if latest_month < 12:
                        st.warning(f"سال جاری ({current_year}) فقط تا ماه {latest_month} کامل است — نتایج تقریبی هستند.")
            # انتخاب نوع نمایش و سال‌ها
            col1, col2 = st.columns([1, 2])
            with col1:
                display_type = st.radio(
                    "نوع تحلیل:",
                    options=["ماهانه", "فصلی", "سالانه"],
                    horizontal=True,
                    key="display_type_select"
                )
            with col2:
                selected_years = st.multiselect(
                    "سال‌های شمسی برای مقایسه:",
                    options=available_years,
                    default=available_years[-3:] if len(available_years) >= 3 else available_years,
                    key="years_multiselect"
                )
            if not selected_years:
                st.warning("لطفاً حداقل یک سال انتخاب کنید.")
            else:
                # فیلتر داده
                df_filtered = df_tab3[df_tab3["سال_شمسی"].isin(selected_years)].copy()
                if df_filtered.empty:
                    st.error("هیچ داده‌ای برای سال(های) انتخابی یافت نشد.")
                else:
                    # توابع کمکی
                    def get_season(month_str):
                        try:
                            m = int(month_str.split("/")[1])
                            return {1: "بهار", 2: "بهار", 3: "بهار", 4: "تابستان", 5: "تابستان", 6: "تابستان",
                                    7: "پاییز", 8: "پاییز", 9: "پاییز", 10: "زمستان", 11: "زمستان", 12: "زمستان"}[m]
                        except:
                            return "نامشخص"
                    df_plot = df_filtered.copy()
                    df_plot["ماه شمسی"] = df_plot["تاریخ"].apply(lambda x: safe_jalali_format(x)[:7] if pd.notnull(x) else "")
                    # آماده‌سازی داده بر اساس نوع نمایش
                    if display_type == "ماهانه":
                        plot_df = df_plot.groupby("ماه شمسی")[monthly_column].sum().reset_index()
                        plot_df = plot_df.sort_values("ماه شمسی")
                        x_col = "ماه شمسی"
                        title_suffix = "ماهانه"
                    elif display_type == "فصلی":
                        df_plot["فصل"] = df_plot["ماه شمسی"].apply(get_season)
                        plot_df = df_plot.groupby("فصل")[monthly_column].sum().reset_index()
                        season_order = ["بهار", "تابستان", "پاییز", "زمستان"]
                        plot_df["فصل"] = pd.Categorical(plot_df["فصل"], categories=season_order, ordered=True)
                        plot_df = plot_df.sort_values("فصل")
                        x_col = "فصل"
                        title_suffix = "فصلی"
                    else: # سالانه
                        plot_df = df_plot.groupby("سال_شمسی")[monthly_column].sum().reset_index()
                        plot_df = plot_df.sort_values("سال_شمسی")
                        x_col = "سال_شمسی"
                        title_suffix = "سالانه"
                    plot_df = plot_df.dropna(subset=[monthly_column])
                    if plot_df.empty:
                        st.warning("داده‌ای برای نمایش وجود ندارد.")
                    else:
                        # رنگ‌بندی بر اساس بیشترین/کمترین
                        max_idx = plot_df[monthly_column].idxmax()
                        min_idx = plot_df[monthly_column].idxmin()
                        plotly_palette = ["#2E8B57" if i == max_idx else "#DC143C" if i == min_idx else "#C74A1B" for i in plot_df.index]
                        # نمودار
                        fig = go.Figure(data=[go.Bar(
                            x=plot_df[x_col],
                            y=plot_df[monthly_column],
                            text=[f"{v:,.0f} {unit}" for v in plot_df[monthly_column]],
                            textposition="outside",
                            marker_color=plotly_palette,
                            hovertemplate=f"<b>%{{x}}</b><br>مصرف: %{{y:,.0f}} {unit}<extra></extra>"
                        )])
                        fig.update_layout(
                            title=f"مصرف {monthly_column} — تحلیل {title_suffix} ({unit})<br><sub>سال‌های: {', '.join(map(str, selected_years))}</sub>",
                            xaxis_title=x_col,
                            yaxis_title=f"مصرف ({unit})",
                            template="plotly_white",
                            xaxis_tickangle=-45,
                            height=600,
                            showlegend=False
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        # KPIهای خلاصه
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("مجموع کل", f"{plot_df[monthly_column].sum():,.0f} {unit}")
                        with col2:
                            st.metric("میانگین", f"{plot_df[monthly_column].mean():,.0f} {unit}")
                        with col3:
                            st.metric("بیشترین", f"{plot_df[monthly_column].max():,.0f} {unit}", delta="بالاترین")
                        with col4:
                            st.metric("کمترین", f"{plot_df[monthly_column].min():,.0f} {unit}", delta="پایین‌ترین")
                        # جدول داده
                        plot_df_display = plot_df.copy()
                        plot_df_display[monthly_column] = plot_df_display[monthly_column].round(0).astype(int)
                        plot_df_display = plot_df_display.rename(columns={monthly_column: f"مصرف ({unit})", x_col: "دوره"})
                        st.dataframe(plot_df_display.style.format({f"مصرف ({unit})": "{:,}"}), use_container_width=True)
                        # دانلود CSV
                        csv = plot_df_display.to_csv(index=False, encoding='utf-8-sig')
                        st.download_button(
                            "دانلود داده‌ها (CSV)",
                            csv,
                            f"تحلیل_{title_suffix}_{monthly_column}.csv",
                            "text/csv",
                            key="download_csv_monthly"
                        )
                        # دانلود PDF حرفه‌ای
                        if st.button("دانلود گزارش کامل PDF", key="download_tab3_pdf"):
                            if IS_CLOUD:
                                st.warning("در محیط کلود فقط CSV قابل دانلود است.")
                            else:
                                with st.spinner("در حال تولید گزارش PDF با کیفیت بالا..."):
                                    buffer = io.BytesIO()
                                    elements = []
                                    # عنوان گزارش
                                    header = f"""
                                    <b>گزارش تحلیل مصرف زمانی — {monthly_column}</b><br/>
                                    نوع تحلیل: {title_suffix} | واحد: {unit}<br/>
                                    سال‌های مورد بررسی: {', '.join(map(str, selected_years))}<br/>
                                    تاریخ گزارش: {safe_jalali_format(pd.Timestamp.today())} | تعداد دوره: {len(plot_df)}
                                    """
                                    elements.append(Paragraph(rtl(header), ParagraphStyle('Normal', fontName=FONT_NAME, fontSize=11, alignment=1, spaceAfter=20)))
                                    # جدول داده
                                    table_data = [["دوره", f"مصرف ({unit})"]] + \
                                                 [[rtl(str(row["دوره"])), rtl(f"{row[f'مصرف ({unit})']:,}")] for _, row in plot_df_display.iterrows()]
                                    table = Table(table_data, colWidths=[250, 200])
                                    table.setStyle(TableStyle([
                                        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#C74A1B")),
                                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                                        ('FONTNAME', (0, 0), (-1, -1), FONT_NAME),
                                        ('FONTSIZE', (0, 0), (-1, -1), 11),
                                        ('GRID', (0, 0), (-1, -1), 0.6, colors.grey),
                                        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor("#fdfdfd")),
                                        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                                    ]))
                                    elements.append(table)
                                    elements.append(Spacer(1, 20))
                                    # تصویر نمودار با کیفیت بالا
                                    try:
                                        img_buf = io.BytesIO()
                                        fig.write_image(img_buf, format="png", engine="kaleido", width=1100, height=650, scale=2.5)
                                        img_buf.seek(0)
                                        elements.append(Image(img_buf, width=520, height=320))
                                    except Exception as e:
                                        st.warning(f"نمودار در PDF اضافه نشد: {e}")
                                    # تولید PDF
                                    generate_pdf(f"تحلیل مصرف {title_suffix} — {monthly_column}", elements, buffer)
                                    buffer.seek(0)
                                    st.success("گزارش PDF با موفقیت تولید شد!")
                                    st.download_button(
                                        label="دانلود گزارش تحلیل زمانی (PDF)",
                                        data=buffer.getvalue(),
                                        file_name=f"تحلیل_{title_suffix}_{monthly_column}_{safe_jalali_format(pd.Timestamp.today())}.pdf",
                                        mime="application/pdf",
                                        key="download_pdf_tab3_final"
                                    )
# ===============================================
# Tab 4: Heatmap مصرف تجهیزات (کاملاً بدون خطا + فارسی + حرفه‌ای)
# ===============================================
with tabs[3]:
    st.subheader("Heatmap تحلیل الگوهای مصرف تجهیزات در تقویم")
    if not selected_equipment:
        st.info("لطفاً از سایدبار حداقل یک تجهیز انتخاب کنید.")
    else:
        heat_col = st.selectbox(
            "انتخاب تجهیز برای نمایش Heatmap:",
            options=selected_equipment,
            index=0,
            key="heatmap_equipment_select"
        )
        unit = get_unit_for_column(filtered_df, heat_col, st.session_state.custom_units)
        view_mode = st.radio(
            "نوع Heatmap:",
            options=["روزانه (روز × ماه)", "ماهانه ماتریسی (ماه × سال)"],
            horizontal=True,
            key="heatmap_view_mode"
        )
        df_hm = filtered_df[[heat_col, "تاریخ"]].dropna(subset=[heat_col]).copy()
        if df_hm.empty:
            st.warning("هیچ داده‌ای برای این تجهیز یافت نشد.")
        else:
            df_hm["تاریخ"] = pd.to_datetime(df_hm["تاریخ"])
            df_hm["سال_شمسی"] = df_hm["تاریخ"].apply(lambda x: JalaliDate(x).year if pd.notnull(x) else None)
            df_hm["روز"] = df_hm["تاریخ"].dt.day
            df_hm["ماه_میلادی"] = df_hm["تاریخ"].dt.to_period("M")
            # تنظیم محدوده رنگ
            vmin, vmax = float(df_hm[heat_col].min()), float(df_hm[heat_col].max())
            col1, col2 = st.columns(2)
            if vmin == vmax:
                st.info(f"مقدار مصرف این تجهیز در کل بازه ثابت است ({vmin:,.1f} {unit})؛ محدوده رنگ قابل تنظیم نیست.")
                user_min, user_max = vmin, vmax if vmax > vmin else vmin + 1
            else:
                with col1:
                    user_min = st.slider("حداقل مقدار رنگ", vmin, vmax, vmin, step=(vmax - vmin) / 50, key="hm_min")
                with col2:
                    user_max = st.slider("حداکثر مقدار رنگ", vmin, vmax, vmax, step=(vmax - vmin) / 50, key="hm_max")
            colorscale = [
                [0.0, "#E3F2FD"], [0.2, "#90CAF9"], [0.4, "#42A5F5"],
                [0.6, "#FFF59D"], [0.8, "#FFCC80"], [1.0, "#FF5252"]
            ]
            fig = go.Figure()
            if view_mode == "روزانه (روز × ماه)":
                pivot = df_hm.pivot_table(
                    index="روز",
                    columns="ماه_میلادی",
                    values=heat_col,
                    aggfunc="mean",
                    fill_value=np.nan
                )
                if pivot.empty:
                    st.error("داده کافی برای Heatmap روزانه وجود ندارد.")
                else:
                    fig.add_trace(go.Heatmap(
                        z=pivot.values,
                        x=[str(col) for col in pivot.columns],
                        y=[str(i) for i in pivot.index],
                        colorscale=colorscale,
                        zmin=user_min,
                        zmax=user_max,
                        colorbar=dict(title=dict(text=unit, side="right")),
                        hovertemplate=(
                            "<b>روز: %{y}</b><br>"
                            "ماه: %{x}<br>"
                            f"مصرف: %{{z:,.1f}} {unit}"
                            "<extra></extra>"
                        )
                    ))
                    fig.update_layout(
                        title=f"Heatmap روزانه مصرف {heat_col} ({unit})<br><sub>هر سلول = میانگین مصرف در آن روز از ماه</sub>",
                        xaxis_title="ماه (میلادی)",
                        yaxis_title="روز ماه",
                        height=max(600, len(pivot) * 22),
                        xaxis_tickangle=-45
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else: # ماهانه ماتریسی
                # نکته: باید از ماه شمسی استفاده شود، نه ماه میلادی (dt.month)،
                # چون در حالت قبلی داده‌ها بر اساس ماه میلادی گروه‌بندی می‌شدند
                # ولی با نام ماه‌های شمسی برچسب‌گذاری می‌شدند و باعث نمایش نادرست داده می‌شد.
                df_hm["ماه_شمسی_عدد"] = df_hm["تاریخ"].apply(lambda x: JalaliDate(x).month if pd.notnull(x) else None)
                pivot_year_month = df_hm.pivot_table(
                    index="ماه_شمسی_عدد",
                    columns="سال_شمسی",
                    values=heat_col,
                    aggfunc="mean",
                    fill_value=np.nan
                )
                if pivot_year_month.empty:
                    st.error("داده کافی برای Heatmap ماهانه وجود ندارد.")
                else:
                    jalali_months = ["فروردین", "اردیبهشت", "خرداد", "تیر", "مرداد", "شهریور",
                                    "مهر", "آبان", "آذر", "دی", "بهمن", "اسفند"]
                    # فقط برچسب ماه‌هایی که واقعاً در داده حضور دارند (طول برابر با ردیف‌های pivot)
                    y_labels = [jalali_months[int(m) - 1] for m in pivot_year_month.index]
                    fig.add_trace(go.Heatmap(
                        z=pivot_year_month.values,
                        x=[str(int(col)) for col in pivot_year_month.columns],
                        y=y_labels,
                        colorscale=colorscale,
                        zmin=user_min,
                        zmax=user_max,
                        colorbar=dict(title=dict(text=unit, side="top")),
                        hovertemplate=(
                            "<b>%{y}</b><br>"
                            "سال: %{x}<br>"
                            f"مصرف: %{{z:,.0f}} {unit}"
                            "<extra></extra>"
                        )
                    ))
                    fig.update_layout(
                        title=f"Heatmap ماهانه مصرف {heat_col} ({unit})<br><sub>هر سلول = میانگین مصرف در آن ماه از سال</sub>",
                        xaxis_title="سال شمسی",
                        yaxis_title="ماه شمسی",
                        height=580
                    )
                    st.plotly_chart(fig, use_container_width=True)
# ===============================================
# Tab 5: خروجی داده‌ها (جدول کامل + دانلود CSV و PDF حرفه‌ای)
# ===============================================
with tabs[4]:
    st.subheader("خروجی کامل داده‌ها (جدول، CSV، PDF)")
    if not selected_equipment:
        st.info("لطفاً از سایدبار حداقل یک تجهیز انتخاب کنید.")
    else:
        # آماده‌سازی داده برای نمایش و دانلود
        df_export = filtered_df[["تاریخ شمسی"] + selected_equipment].copy()
        # نمایش جدول با واحد (فقط برای کاربر)
        df_display = df_export.copy()
        for col in selected_equipment:
            unit = get_unit_for_column(filtered_df, col, st.session_state.custom_units)
            df_display[col] = df_display[col].apply(lambda x: f"{x:,.2f} {unit}" if pd.notnull(x) and x != '' else "-")
        st.markdown(f"**تعداد رکورد:** {len(df_display):,} | **تجهیزات انتخاب‌شده:** {len(selected_equipment)}")
        st.dataframe(df_display, use_container_width=True, height=500)
        # دانلود CSV (با واحد در هدر + بدون NaN)
        df_csv = df_export.copy()
        df_csv.columns = ["تاریخ شمسی"] + [f"{col} ({get_unit_for_column(filtered_df, col, st.session_state.custom_units)})" for col in selected_equipment]
        csv_bytes = df_csv.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="دانلود داده‌ها به صورت CSV (برای اکسل)",
                data=csv_bytes,
                file_name=f"داده_کامل_{safe_jalali_format(pd.Timestamp.today())}.csv",
                mime="text/csv",
                key="download_csv_full"
            )
        # دانلود PDF حرفه‌ای (تا ۲۰۰۰ رکورد + صفحه‌بندی هوشمند)
        with col2:
            pdf_requested = st.button("دانلود گزارش کامل داده‌ها به صورت PDF", key="download_pdf_full")
        if pdf_requested:
            if IS_CLOUD:
                st.warning("در محیط کلود، فقط CSV قابل دانلود است.")
            else:
                with st.spinner("در حال تولید PDF حرفه‌ای (ممکن است چند ثانیه طول بکشد)..."):
                    buffer = io.BytesIO()
                    elements = []
                    # عنوان گزارش
                    header = f"""
                    <b>گزارش کامل داده‌های پایش انرژی</b><br/>
                    بازه زمانی: {safe_jalali_format(start_date)} تا {safe_jalali_format(end_date)}<br/>
                    تعداد رکورد: {len(df_export):,} | تعداد تجهیزات: {len(selected_equipment)}<br/>
                    تاریخ گزارش: {safe_jalali_format(pd.Timestamp.today())} | تهیه‌شده توسط داشبورد SIMIDCO
                    """
                    elements.append(Paragraph(rtl(header), ParagraphStyle('Normal', fontName=FONT_NAME, fontSize=12, alignment=1, spaceAfter=30)))
                    # جدول با هدر واحددار
                    headers = ["ردیف", "تاریخ شمسی"] + [f"{col}<br/>({get_unit_for_column(filtered_df, col, st.session_state.custom_units)})" for col in selected_equipment]
                    data = [headers]
                    # فقط ۲۰۰۰ رکورد اول (برای جلوگیری از PDF خیلی بزرگ)
                    max_rows = 2000
                    # از enumerate برای شماره‌گذاری صحیح و پیوسته ردیف‌ها استفاده می‌شود
                    # (ایندکس اصلی df_export پیوسته نیست، چون از فیلتر تاریخ روی دیتافریم اصلی به‌دست آمده)
                    for row_num, (idx, row) in enumerate(df_export.head(max_rows).iterrows(), start=1):
                        row_data = [str(row_num), rtl(row["تاریخ شمسی"])]
                        for col in selected_equipment:
                            val = row[col]
                            formatted = f"{val:,.2f}" if pd.notnull(val) and val != '' else "-"
                            row_data.append(rtl(formatted))
                        data.append(row_data)
                    # تنظیم عرض ستون‌ها
                    col_widths = [50, 100] + [100] * len(selected_equipment)
                    table = Table(data, colWidths=col_widths, repeatRows=1)
                    table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#C74A1B")),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                        ('FONTNAME', (0, 0), (-1, -1), FONT_NAME),
                        ('FONTSIZE', (0, 0), (-1, -1), 9),
                        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor("#fdfdfd")),
                        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                        ('INNERGRID', (0, 0), (-1, -1), 0.25, colors.lightgrey),
                        ('BOX', (0, 0), (-1, -1), 1, colors.black),
                        ('ROWBACKGROUNDS', (1, 0), (-1, -1), [colors.white, colors.HexColor("#f5f5f5")]),
                    ]))
                    elements.append(table)
                    # اگر داده بیشتر از ۲۰۰۰ بود، یادداشت
                    if len(df_export) > max_rows:
                        note = f"<i>توجه: فقط {max_rows:,} رکورد اول نمایش داده شد. برای داده کامل از CSV استفاده کنید.</i>"
                        elements.append(Paragraph(rtl(note), ParagraphStyle('Normal', fontName=FONT_NAME, fontSize=10, alignment=1, spaceBefore=20)))
                    # لوگو یا فوتر (اختیاری)
                    footer = "<i>داشبورد پایش برق کنسانتره — توسعه فراگیر سناباد</i>"
                    elements.append(Paragraph(rtl(footer), ParagraphStyle('Normal', fontName=FONT_NAME, fontSize=9, alignment=1, spaceBefore=30)))
                    # تولید PDF
                    generate_pdf("گزارش کامل داده‌های پایش انرژی", elements, buffer)
                    buffer.seek(0)
                    st.success(f"PDF با {min(len(df_export), max_rows):,} رکورد با موفقیت تولید شد!")
                    st.download_button(
                        label="دانلود گزارش کامل داده‌ها (PDF)",
                        data=buffer.getvalue(),
                        file_name=f"داده_کامل_پایش_انرژی_{safe_jalali_format(pd.Timestamp.today())}.pdf",
                        mime="application/pdf",
                        key="download_pdf_full_final"
                    )
# ----------- Tab6: پیش‌بینی -----------
# ===============================================
# Tab 6: پیش‌بینی هوشمند مصرف تجهیزات — ۱۰۰٪ شمسی + حرفه‌ای
# ===============================================
with tabs[5]:
    st.subheader("# پیش‌بینی هوشمند مصرف تجهیزات در آینده")
    if not selected_equipment:
        st.info("لطفاً از سایدبار حداقل یک تجهیز انتخاب کنید.")
    else:
        forecast_col = st.selectbox(
            "انتخاب تجهیز برای پیش‌بینی:",
            options=selected_equipment,
            index=0,
            key="forecast_equipment_select"
        )
        unit = get_unit_for_column(filtered_df, forecast_col, st.session_state.custom_units)
        # تنظیمات پیش‌بینی
        col1, col2 = st.columns(2)
        with col1:
            horizon = st.slider("افق پیش‌بینی (روز):", 7, 90, 30, step=5, key="forecast_horizon")
        with col2:
            granularity = st.radio(
                "نمایش داده‌ها:",
                ["روزانه", "ماهانه", "سالیانه"],
                horizontal=True,
                index=0,
                key="forecast_granularity"
            )
        # آماده‌سازی داده
        df_pred = filtered_df[["تاریخ", forecast_col]].dropna().copy()
        if len(df_pred) < 15:
            st.error("داده کافی برای پیش‌بینی وجود ندارد (حداقل ۱۵ رکورد نیاز است).")
        else:
            df_pred = df_pred.sort_values("تاریخ").reset_index(drop=True)
            df_pred["روز_عددی"] = (df_pred["تاریخ"] - df_pred["تاریخ"].min()).dt.days
            # مدل رگرسیون خطی
            X = df_pred[["روز_عددی"]].values
            y = df_pred[forecast_col].values
            model = LinearRegression()
            model.fit(X, y)
            # پیش‌بینی آینده
            last_day = df_pred["روز_عددی"].max()
            future_days = np.arange(last_day + 1, last_day + 1 + horizon).reshape(-1, 1)
            future_pred = model.predict(future_days)
            future_dates = pd.date_range(df_pred["تاریخ"].max() + pd.Timedelta(days=1), periods=horizon)
            # تبدیل تاریخ به شمسی — کاملاً درست
            def to_jalali_period(date, period_type):
                jd = JalaliDate(date)
                if period_type == "روزانه":
                    return jd.strftime("%Y/%m/%d")
                elif period_type == "ماهانه":
                    return jd.strftime("%Y/%m")
                else: # سالیانه
                    return str(jd.year)
            # داده واقعی (تاریخ شمسی)
            if granularity == "روزانه":
                hist_x = df_pred["تاریخ"].apply(lambda x: to_jalali_period(x, "روزانه"))
                hist_y = df_pred[forecast_col]
            elif granularity == "ماهانه":
                df_pred["ماه_شمسی"] = df_pred["تاریخ"].apply(lambda x: to_jalali_period(x, "ماهانه"))
                monthly = df_pred.groupby("ماه_شمسی")[forecast_col].sum().reset_index()
                hist_x = monthly["ماه_شمسی"]
                hist_y = monthly[forecast_col]
                # پیش‌بینی ماهانه (تقریبی — هر ۳۰ روز یک نقطه)
                future_x = [to_jalali_period(d, "ماهانه") for d in future_dates[::30]]
                future_pred_monthly = np.cumsum([future_pred[i:i+30].sum() for i in range(0, len(future_pred), 30)])
                future_pred = future_pred_monthly
            else: # سالیانه
                df_pred["سال_شمسی"] = df_pred["تاریخ"].apply(lambda x: to_jalali_period(x, "سالیانه"))
                yearly = df_pred.groupby("سال_شمسی")[forecast_col].sum().reset_index()
                hist_x = yearly["سال_شمسی"]
                hist_y = yearly[forecast_col]
                future_x = [str(JalaliDate(future_dates[-1]).year)]
                future_pred = [future_pred.sum()]
            # آینده (شمسی)
            future_x = [to_jalali_period(d, granularity) for d in future_dates[::(30 if granularity == "ماهانه" else 365 if granularity == "سالیانه" else 1)]]
            # نمودار نهایی
            fig = go.Figure()
            # داده واقعی
            fig.add_trace(go.Scatter(
                x=hist_x,
                y=hist_y,
                mode="lines+markers",
                name="مصرف واقعی",
                line=dict(color="#1A1A1A", width=4),
                marker=dict(size=10)
            ))
            # پیش‌بینی
            fig.add_trace(go.Scatter(
                x=future_x,
                y=future_pred,
                mode="lines+markers",
                name=f"پیش‌بینی {horizon} روز آینده",
                line=dict(color="#C74A1B", width=5, dash="dash"),
                marker=dict(size=12, symbol="diamond", color="#C74A1B", line=dict(width=2, color="white")),
                hovertemplate=f"<b>پیش‌بینی</b><br>تاریخ: %{{x}}<br>مصرف: %{{y:,.0f}} {unit}<extra></extra>"
            ))
            # شیب و R²
            slope = model.coef_[0]
            r2 = model.score(X, y)
            fig.update_layout(
                title=f"پیش‌بینی مصرف {forecast_col} ({unit})<br>"
                      f"<sub>شیب روند: {slope:+.2f} {unit}/روز | دقت مدل (R²): {r2:.3f} | پیش‌بینی تا: {future_x[-1]}</sub>",
                xaxis_title="تاریخ (شمسی)",
                yaxis_title=f"مصرف ({unit})",
                hovermode="x unified",
                template="plotly_white",
                height=650,
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
                margin=dict(t=140)
            )
            st.plotly_chart(fig, use_container_width=True)
            # تحلیل هوشمند
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("مصرف فعلی (آخرین)", f"{df_pred[forecast_col].iloc[-1]:,.0f} {unit}")
            with col2:
                predicted = future_pred[-1]
                st.metric("پیش‌بینی پایان دوره", f"{predicted:,.0f} {unit}", delta=f"{(predicted - df_pred[forecast_col].iloc[-1]):+,.0f}")
            with col3:
                trend = "افزایشی" if slope > 0.1 else "کاهشی" if slope < -0.1 else "پایدار"
                st.metric("روند کلی", trend, delta=f"{slope:+.2f} {unit}/روز")
            if slope > 0.5:
                st.warning(f"روند صعودی قوی — مصرف {forecast_col} در حال افزایش است!")
            elif slope < -0.5:
                st.success(f"روند نزولی — مصرف {forecast_col} در حال کاهش است — عملکرد عالی")
            else:
                st.info("مصرف تقریباً پایدار است")
            # دانلود PDF حرفه‌ای
            if st.button("دانلود گزارش پیش‌بینی (PDF)", key="download_forecast_pdf"):
                buffer = io.BytesIO()
                elements = []
                header = f"""
                <b>گزارش پیش‌بینی هوشمند مصرف {forecast_col}</b><br/>
                واحد: {unit} | افق: {horizon} روز | دقت مدل (R²): {r2:.3f}<br/>
                شیب روند: {slope:+.2f} {unit}/روز | آخرین داده: {safe_jalali_format(df_pred["تاریخ"].max())} | پیش‌بینی تا: {future_x[-1]}<br/>
                تاریخ گزارش: {safe_jalali_format(pd.Timestamp.today())}
                """
                elements.append(Paragraph(rtl(header), ParagraphStyle('Title', fontName=FONT_NAME, fontSize=14, alignment=1, spaceAfter=30)))
                # تحلیل
                analysis = f"""
                <b>تحلیل خودکار:</b><br/>
                • مصرف فعلی: <b>{df_pred[forecast_col].iloc[-1]:,.0f}</b> {unit}<br/>
                • پیش‌بینی پایان دوره: <b>{predicted:,.0f}</b> {unit}<br/>
                • تغییر پیش‌بینی‌شده: <b>{(predicted - df_pred[forecast_col].iloc[-1]):+,.0f}</b> {unit}<br/>
                • روند: <b>{trend}</b> — مدل با دقت {r2:.1%} داده‌های گذشته را توضیح می‌دهد
                """
                elements.append(Paragraph(rtl(analysis), ParagraphStyle('Normal', fontName=FONT_NAME, fontSize=11, leading=16, spaceAfter=20)))
                # تصویر
                try:
                    img = io.BytesIO()
                    fig.write_image(img, format="png", engine="kaleido", width=1200, height=700, scale=3)
                    img.seek(0)
                    elements.append(Image(img, width=550, height=320))
                except:
                    pass
                generate_pdf(f"پیش‌بینی {forecast_col}", elements, buffer)
                buffer.seek(0)
                st.download_button(
                    "دانلود گزارش پیش‌بینی",
                    buffer.getvalue(),
                    f"پیش_بینی_{forecast_col}_{safe_jalali_format(pd.Timestamp.today())}.pdf",
                    "application/pdf"
                )

# ===============================================
# Tab 7: KPI پیشرفته — شاخص‌های کلیدی عملکرد انرژی (کاملاً بدون خطا + تاریخ شمسی)
# ===============================================
with tabs[6]:
    st.subheader("# KPI پیشرفته — شاخص‌های کلیدی عملکرد انرژی")
    if not selected_equipment:
        st.info("لطفاً از سایدبار حداقل یک تجهیز انتخاب کنید.")
    else:
        kpi_columns = st.multiselect(
            "انتخاب تجهیزات برای تحلیل KPI:",
            options=selected_equipment,
            default=selected_equipment[:4] if len(selected_equipment) >= 4 else selected_equipment,
            key="kpi_equipment_select"
        )
        if not kpi_columns:
            st.warning("حداقل یک تجهیز انتخاب کنید.")
        else:
            # -------------------- 1. کارت‌های KPI خام --------------------
            st.markdown("### کارت‌های KPI (مصرف واقعی)")
            cols = st.columns(len(kpi_columns))
            for idx, col in enumerate(kpi_columns):
                with cols[idx % len(cols)]:
                    unit = get_unit_for_column(filtered_df, col, st.session_state.custom_units)
                    total = filtered_df[col].sum()
                    avg = filtered_df[col].mean()
                    max_val = filtered_df[col].max()
                    st.metric(
                        label=col,
                        value=f"{total:,.0f} {unit}",
                        delta=f"میانگین: {avg:,.1f} | حداکثر: {max_val:,.0f}"
                    )
            # -------------------- 2. تبدیل به GJ --------------------
            st.markdown("### تبدیل به گیگاژول (GJ) — معادل انرژی")
            if 'manual_factors' not in st.session_state:
                st.session_state.manual_factors = {}
            gj_results = []
            for col in kpi_columns:
                unit = get_unit_for_column(filtered_df, col, st.session_state.custom_units)
                default_factor = 3.6 if "برق" in col else 0.038 if "گاز" in col else 0.038
                factor = st.number_input(
                    f"فاکتور تبدیل {col} ({unit} → GJ)",
                    value=float(st.session_state.manual_factors.get(col, default_factor)),
                    step=0.001,
                    format="%.6f",
                    key=f"factor_{col}"
                )
                st.session_state.manual_factors[col] = factor
                total_raw = filtered_df[col].sum()
                total_gj = total_raw * factor
                gj_results.append({
                    "تجهیز": col,
                    "مصرف اصلی": f"{total_raw:,.2f} {unit}",
                    "فاکتور": factor,
                    "معادل GJ": round(total_gj, 2)
                })
            gj_df = pd.DataFrame(gj_results)
            st.dataframe(gj_df, use_container_width=True)
            total_gj = gj_df["معادل GJ"].sum()
            st.metric("مجموع کل انرژی (GJ)", f"{total_gj:,.2f}", delta=f"از {len(kpi_columns)} حامل")
            # -------------------- 3. EnPI (امن و بدون خطا) --------------------
            st.markdown("### EnPI — شاخص عملکرد انرژی (kWh/تن تولید)")
            prod_cols = [c for c in filtered_df.columns if any(x in c.lower() for x in ["تولید", "production", "تن"])]
            selected_prod = st.selectbox("انتخاب ستون تولید (تن):", options=["انتخاب کنید"] + prod_cols, key="enpi_prod_col")
            avg_enpi = None # مقدار پیش‌فرض
            if selected_prod != "انتخاب کنید" and selected_prod in filtered_df.columns:
                total_prod = filtered_df[selected_prod].sum()
                if total_prod > 0:
                    total_kwh = 0
                    for col in kpi_columns:
                        unit = get_unit_for_column(filtered_df, col, st.session_state.custom_units)
                        total_kwh += filtered_df[col].sum() * (1000 if unit == "MWh" else 1)
                    avg_enpi = total_kwh / total_prod
                    st.metric("EnPI کل سیستم", f"{avg_enpi:.3f} kWh/تن", delta="هدف: کمتر از ۵")
                else:
                    st.error("مجموع تولید صفر است.")
            else:
                st.info("ستون تولید را انتخاب کنید تا EnPI محاسبه شود.")
            # -------------------- 4. سهم انرژی (با تاریخ شمسی کامل) --------------------
            st.markdown("### سهم انرژی (GJ) — ماه‌های اخیر (تقویم شمسی)")

            # تبدیل تاریخ به ماه شمسی (روی یک کپی محلی، تا دیتافریم مشترک بین تب‌ها دستکاری نشود)
            df_tab7 = filtered_df.copy()
            df_tab7["ماه_شمسی"] = df_tab7["تاریخ"].apply(
                lambda x: safe_jalali_format(x)[:7] if pd.notnull(x) else None
            )
            available_months = sorted([m for m in df_tab7["ماه_شمسی"].dropna().unique() if m])

            if len(available_months) < 3:
                st.warning("داده کافی برای نمایش سهم ماهانه وجود ندارد.")
            else:
                # پیش‌فرض: ۶ ماه اخیر
                default_months = available_months[-6:]
                selected_months = st.multiselect(
                    "انتخاب ماه‌های شمسی:",
                    options=available_months,
                    default=default_months,
                    key="pie_months_shamsi"
                )
                if selected_months:
                    df_pie = df_tab7[df_tab7["ماه_شمسی"].isin(selected_months)].copy()
                    pie_data = []
                    for col in kpi_columns:
                        total = df_pie[col].sum()
                        factor = st.session_state.manual_factors.get(col, 3.6)
                        gj = total * factor
                        pie_data.append({"حامل": col, "GJ": gj})
                    pie_df = pd.DataFrame(pie_data)
                    total_gj_pie = pie_df["GJ"].sum()
                    if total_gj_pie > 0:
                        pie_df["سهم (%)"] = (pie_df["GJ"] / total_gj_pie * 100).round(1)
                        fig_pie = px.pie(
                            pie_df,
                            names="حامل",
                            values="GJ",
                            title=f"سهم انرژی در {len(selected_months)} ماه اخیر (تقویم شمسی)",
                            hole=0.4
                        )
                        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                        st.plotly_chart(fig_pie, use_container_width=True)
                    else:
                        st.warning("⚠️ مجموع انرژی (GJ) برای ماه‌های انتخابی صفر است — نمودار سهم قابل رسم نیست.")
            # -------------------- 5. دانلود PDF کامل (بدون خطا) --------------------
            if st.button("دانلود گزارش کامل KPI پیشرفته (PDF)", key="download_kpi_full_pdf"):
                buffer = io.BytesIO()
                elements = []
                # عنوان
                title_text = f"""
                <b>گزارش جامع KPI پیشرفته انرژی — SIMIDCO</b><br/>
                بازه: {safe_jalali_format(start_date)} تا {safe_jalali_format(end_date)} | تاریخ گزارش: {safe_jalali_format(pd.Timestamp.today())}
                """
                elements.append(Paragraph(rtl(title_text), ParagraphStyle('Title', fontName=FONT_NAME, fontSize=16, alignment=1, spaceAfter=30)))
                # جدول GJ
                table_data = [["تجهیز", "معادل GJ", "فاکتور"]]
                for row in gj_results:
                    table_data.append([rtl(row["تجهیز"]), rtl(f"{row['معادل GJ']:,}"), rtl(f"{row['فاکتور']:.6f}")])
                table = Table(table_data)
                table.setStyle(TableStyle([
                    ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#2E7D32")),
                    ('TEXTCOLOR', (0,0), (-1,0), colors.white),
                    ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                    ('FONTNAME', (0,0), (-1,-1), FONT_NAME),
                    ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
                ]))
                elements.append(table)
                # EnPI (امن)
                enpi_text = f"EnPI کل: <b>{avg_enpi:.2f}</b> kWh/تن (هدف: کمتر از ۵)" if avg_enpi else "EnPI محاسبه نشده (ستون تولید انتخاب نشده)"
                elements.append(Paragraph(rtl(enpi_text), ParagraphStyle('Normal', fontName=FONT_NAME, alignment=1, spaceAfter=20)))
                # نمودار Pie (اگر وجود داشته باشه)
                if 'fig_pie' in locals():
                    try:
                        img = io.BytesIO()
                        fig_pie.write_image(img, format="png", engine="kaleido", width=1000, height=600, scale=3)
                        img.seek(0)
                        elements.append(Image(img, width=500, height=300))
                    except:
                        pass
                generate_pdf("گزارش KPI پیشرفته انرژی", elements, buffer)
                buffer.seek(0)
                st.download_button(
                    "دانلود گزارش کامل KPI (PDF)",
                    buffer.getvalue(),
                    f"KPI_پیشرفته_{safe_jalali_format(pd.Timestamp.today())}.pdf",
                    "application/pdf"
                )
# ----------- Tab8: تحلیل روند تغییرات -----------
with tabs[7]:
    st.subheader("📈 تغییرات درصدی نسبت به دوره قبلی")
    if not selected_equipment:
        st.info("لطفاً از سایدبار حداقل یک تجهیز انتخاب کنید.")
    else:
        selected_col = st.selectbox("🔌 انتخاب تجهیز:", selected_equipment)
        unit = get_unit_for_column(filtered_df, selected_col, st.session_state.custom_units)  # 👈 تغییر: واحد تجهیز
        period_type = st.radio("⏱️ بازه زمانی:", ["روزانه", "ماهانه", "سالیانه"])
        if selected_col:
            df_change = filtered_df.copy()
            # نکته: گروه‌بندی بر اساس تقویم شمسی انجام می‌شود (نه میلادی)
            # تا با تحلیل ماهانه/سالانه تب «ماهانه» هم‌خوان باشد.
            if period_type == "روزانه":
                df_change["period"] = df_change["تاریخ"].apply(safe_jalali_format)
            elif period_type == "ماهانه":
                df_change["period"] = df_change["تاریخ"].apply(lambda x: safe_jalali_format(x)[:7] if pd.notnull(x) else None)
            elif period_type == "سالیانه":
                df_change["period"] = df_change["تاریخ"].apply(lambda x: JalaliDate(x).year if pd.notnull(x) else None)
            df_change = df_change.dropna(subset=["period"])
            df_change = df_change.groupby("period")[selected_col].sum().reset_index()
            df_change = df_change.sort_values("period")
            df_change["تاریخ نمایش"] = df_change["period"].astype(str)  # برچسب شمسی (روز/ماه/سال بسته به بازه انتخابی)
            df_change["درصد تغییر"] = df_change[selected_col].pct_change() * 100
            df_change.replace([np.inf, -np.inf], np.nan, inplace=True)
            df_change["درصد تغییر"] = df_change["درصد تغییر"].fillna(0).astype(float)
            if not df_change.empty:
                st.markdown("📋 **جدول تغییرات درصدی**")
                # 👈 تغییر: نمایش با واحد
                df_change_display = df_change.copy()
                df_change_display[selected_col] = df_change_display[selected_col].astype(str) + f" {unit}"
                st.dataframe(
                    df_change_display[["تاریخ نمایش", selected_col, "درصد تغییر"]].round(2),
                    use_container_width=True
                )
                fig_line = px.line(
                    df_change,
                    x="تاریخ نمایش",
                    y="درصد تغییر",
                    title=f"📈 تغییرات درصدی {selected_col} ({period_type})",
                    markers=True,
                    template="plotly_white"
                )
                fig_line.update_layout(yaxis_title="درصد تغییر (%)", xaxis_title="تاریخ")
                st.plotly_chart(fig_line, use_container_width=True)
                measures = ["relative"] * len(df_change)
                measures[-1] = "total"
                y_values = df_change["درصد تغییر"].tolist()
                x_values = df_change["تاریخ نمایش"].tolist()
                fig_wf = go.Figure(go.Waterfall(
                    name="درصد تغییر",
                    orientation="v",
                    measure=measures,
                    x=x_values,
                    y=y_values,
                    decreasing=dict(marker=dict(color="red")),
                    increasing=dict(marker=dict(color="green")),
                    totals=dict(marker=dict(color="blue")),
                    text=[f"{v:.2f}%" for v in y_values],
                    textposition="outside",
                ))
                fig_wf.update_layout(
                    title=f"💧 نمودار Waterfall تغییرات درصدی {selected_col} ({period_type})",
                    yaxis_title="درصد تغییر (%)",
                    xaxis_title="تاریخ"
                )
                st.plotly_chart(fig_wf, use_container_width=True)
                idx_max = df_change["درصد تغییر"].idxmax()
                idx_min = df_change["درصد تغییر"].idxmin()
                max_increase = df_change.loc[idx_max]
                max_decrease = df_change.loc[idx_min]
                st.markdown(
                    f"**نتیجه‌گیری:**\n\n"
                    f"🔺 بیشترین افزایش مصرف **{selected_col}**: "
                    f"**{max_increase['درصد تغییر']:.2f}%** در دوره **{max_increase['تاریخ نمایش']}**\n\n"
                    f"🔻 بیشترین کاهش مصرف **{selected_col}**: "
                    f"**{max_decrease['درصد تغییر']:.2f}%** در دوره **{max_decrease['تاریخ نمایش']}**"
                )
                # خروجی PDF برای Tab8
                if st.button("⬇️ دانلود PDF Tab8", key="download_tab8_pdf_v1"):  # 👈 فیکس: unique key
                    buffer = io.BytesIO()
                    elements = []

                    # 👈 تغییر: data با واحد
                    df_change_pdf = df_change.copy()
                    df_change_pdf[selected_col] = df_change_pdf[selected_col].round(2).astype(str) + f" {unit}"
                    df_change_pdf["درصد تغییر"] = df_change_pdf["درصد تغییر"].round(2).astype(str) + " %"
                    data = [df_change_pdf[["تاریخ نمایش", selected_col, "درصد تغییر"]].columns.tolist()] + df_change_pdf[["تاریخ نمایش", selected_col, "درصد تغییر"]].fillna(0).values.tolist()  # 👈 فیکس: fillna(0)

                    # ترجمه هدرها اگر انگلیسی
                    translations_local = {
                        "تاریخ نمایش": "Display Date",
                        selected_col: "Consumption",
                        "درصد تغییر": "Percent Change"
                    }
                    if not USE_PERSIAN and data and isinstance(data[0], list) and len(data[0]) >= 3:
                        data[0][0] = translations_local.get(data[0][0], data[0][0])
                        data[0][1] = translations_local.get(data[0][1], data[0][1])
                        data[0][2] = translations_local.get(data[0][2], data[0][2])

                    # Reshape متن‌ها برای RTL اگر فارسی (فقط رشته‌ها)
                    if USE_PERSIAN:
                        try:
                            for row in data:
                                for i, cell in enumerate(row):
                                    if isinstance(cell, str):
                                        row[i] = get_display(arabic_reshaper.reshape(cell))
                        except ImportError:
                            st.warning("برای RTL در جدول، arabic-reshaper و python-bidi رو نصب کن.")

                    table = Table(data)
                    table.setStyle(TableStyle([
                        ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),  # پس‌زمینه عنوان
                        ('TEXTCOLOR', (0,0), (-1,0), colors.black),  # متن عنوان
                        ('GRID', (0,0), (-1,-1), 0.5, colors.lightgrey),  # 👈 grid کم‌رنگ (نه مشکی)
                        ('ALIGN', (0,0), (-1,-1), 'CENTER')
                    ]))
                    elements.append(table)

                    # تولید تصویر با کیفیت بهتر (نیاز به kaleido: pip install kaleido)
                    img_buf1 = io.BytesIO()
                    fig_line.write_image(img_buf1, format='png', width=800, height=400, scale=2)
                    img_buf1.seek(0)
                    elements.append(Image(img_buf1, width=500, height=300))

                    img_buf2 = io.BytesIO()
                    fig_wf.write_image(img_buf2, format='png', width=800, height=400, scale=2)
                    img_buf2.seek(0)
                    elements.append(Image(img_buf2, width=500, height=300))

                    title = f"تحلیل روند تغییرات ({unit})"
                    if not USE_PERSIAN:
                        title = translations.get(title, title)
                    generate_pdf(title, elements, buffer)

                    # 👈 دانلود با getvalue() برای اطمینان
                    pdf_data = buffer.getvalue()
                    st.download_button(
                        label="دانلود PDF",
                        data=pdf_data,
                        file_name=f"روند_تغییرات_{selected_col}_{safe_jalali_format(pd.Timestamp.today())}.pdf",
                        mime="application/pdf"
                    )

                    # چک فونت
                    if not USE_PERSIAN:
                        st.warning("⚠️ فونت فارسی پیدا نشد. PDF به انگلیسی تولید شد.")
            else:
                st.warning("📭 داده کافی برای رسم نمودار وجود ندارد.")
# ===============================================
# Tab 9: ML پیش‌بینی
# ===============================================
# ----------- Tab9: پیش‌بینی ML -----------
with tabs[8]:
    st.subheader("🤖 پیش‌بینی مصرف تجهیزات با Machine Learning")
    selected_cols = st.multiselect("🔌 انتخاب تجهیزات (ML):", selected_equipment, key="ml_multiselect")
    if selected_cols:
        # نکته: این متغیرها با نام مجزا (ml_ prefix) تعریف می‌شوند تا با start_date/end_date
        # سراسری (که در سایدبار تنظیم شده و در PDF بقیه تب‌ها استفاده می‌شود) تداخل نکنند.
        ml_start_date = st.date_input("📅 شروع بازه (ML)", value=filtered_df['تاریخ'].min(), key="ml_start_date")
        ml_end_date = st.date_input("📅 پایان بازه (ML)", value=filtered_df['تاریخ'].max(), key="ml_end_date")
        df_ml = filtered_df[(filtered_df['تاریخ'] >= pd.to_datetime(ml_start_date)) &
                            (filtered_df['تاریخ'] <= pd.to_datetime(ml_end_date))].copy()
        error_table = []
        for col in selected_cols:
            unit = get_unit_for_column(filtered_df, col, st.session_state.custom_units)  # 👈 تغییر: واحد تجهیز
            st.markdown(f"### 🔹 پیش‌بینی {col}")
            ts = df_ml[['تاریخ', col]].rename(columns={'تاریخ': 'ds', col: 'y'}).dropna()
            ts['ds'] = pd.to_datetime(ts['ds'])
            ts = ts.sort_values('ds').reset_index(drop=True)
            if len(ts) < 4:
                st.warning(f"داده برای {col} کافی نیست (حداقل 4 رکورد لازم است).")
                continue
            train_size = int(len(ts) * 0.8)
            if train_size < 1:
                train_size = len(ts) - 1
            train_df = ts.iloc[:train_size].copy().reset_index(drop=True)
            test_df = ts.iloc[train_size:].copy().reset_index(drop=True)
            preds_prophet_test = pd.DataFrame(columns=['ds', 'Predicted', 'Lower', 'Upper'])
            preds_arima_test = pd.DataFrame(columns=['ds', 'Predicted', 'Lower', 'Upper'])
            preds_exp_test = pd.DataFrame(columns=['ds', 'Predicted', 'Lower', 'Upper'])
            df_pred_future_prophet = pd.DataFrame(columns=['ds', 'Predicted', 'Lower', 'Upper'])
            df_pred_future_arima = pd.DataFrame(columns=['ds', 'Predicted', 'Lower', 'Upper'])
            df_pred_future_exp = pd.DataFrame(columns=['ds', 'Predicted', 'Lower', 'Upper'])
            if PROPHET_AVAILABLE:  # 👈 فیکس: چک خارج از loop
                try:
                    m = Prophet(daily_seasonality=True)
                    m.fit(train_df)
                    future_test = test_df[['ds']].copy()
                    forecast_test = m.predict(future_test)
                    preds_prophet_test = forecast_test[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].rename(
                        columns={'yhat': 'Predicted', 'yhat_lower': 'Lower', 'yhat_upper': 'Upper'}
                    )
                    merged_p = test_df.merge(preds_prophet_test, on='ds', how='left')
                    valid_p = merged_p.dropna(subset=['y', 'Predicted'])
                    mae_prophet = valid_p['y'].sub(valid_p['Predicted']).abs().mean() if len(valid_p) > 0 else float('nan')
                    rmse_prophet = ((valid_p['y'] - valid_p['Predicted'])**2).mean()**0.5 if len(valid_p) > 0 else float('nan')
                except Exception as e:
                    mae_prophet = rmse_prophet = float('nan')
                    st.error(f"⚠️ خطا در اجرای Prophet برای {col}: {e}")
            else:
                mae_prophet = rmse_prophet = float('nan')
            if STATSMODELS_AVAILABLE:  # 👈 فیکس: چک خارج از loop
                try:
                    ts_arima_train = train_df.set_index('ds')['y']
                    arima_model = ARIMA(ts_arima_train, order=(1,1,1))
                    arima_fit = arima_model.fit()
                    if len(test_df) > 0:
                        forecast_test_a = arima_fit.get_forecast(steps=len(test_df))
                        conf_int = forecast_test_a.conf_int(alpha=0.05)
                        preds_arima_test = pd.DataFrame({
                            'ds': test_df['ds'].values,
                            'Predicted': forecast_test_a.predicted_mean.values,
                            'Lower': conf_int.iloc[:, 0].values,
                            'Upper': conf_int.iloc[:, 1].values
                        })
                        merged_a = test_df.merge(preds_arima_test, on='ds', how='left')
                        valid_a = merged_a.dropna(subset=['y', 'Predicted'])
                        mae_arima = valid_a['y'].sub(valid_a['Predicted']).abs().mean() if len(valid_a) > 0 else float('nan')
                        rmse_arima = ((valid_a['y'] - valid_a['Predicted'])**2).mean()**0.5 if len(valid_a) > 0 else float('nan')
                    else:
                        mae_arima = rmse_arima = float('nan')
                except Exception as e:
                    mae_arima = rmse_arima = float('nan')
                    st.error(f"⚠️ خطا در اجرای ARIMA برای {col}: {e}")
                try:
                    ts_exp_train = train_df.set_index('ds')['y']
                    exp_model = ExponentialSmoothing(ts_exp_train, trend='add', seasonal=None, damped_trend=True)
                    exp_fit = exp_model.fit()
                    if len(test_df) > 0:
                        forecast_test_e = exp_fit.forecast(steps=len(test_df))
                        std_err = np.std(exp_fit.resid)
                        preds_exp_test = pd.DataFrame({
                            'ds': test_df['ds'].values,
                            'Predicted': forecast_test_e.values,
                            'Lower': forecast_test_e.values - 1.96 * std_err,
                            'Upper': forecast_test_e.values + 1.96 * std_err
                        })
                        merged_e = test_df.merge(preds_exp_test, on='ds', how='left')
                        valid_e = merged_e.dropna(subset=['y', 'Predicted'])
                        mae_exp = valid_e['y'].sub(valid_e['Predicted']).abs().mean() if len(valid_e) > 0 else float('nan')
                        rmse_exp = ((valid_e['y'] - valid_e['Predicted'])**2).mean()**0.5 if len(valid_e) > 0 else float('nan')
                    else:
                        mae_exp = rmse_exp = float('nan')
                except Exception as e:
                    mae_exp = rmse_exp = float('nan')
                    st.error(f"⚠️ خطا در اجرای Exponential Smoothing برای {col}: {e}")
            else:
                mae_arima = rmse_arima = mae_exp = rmse_exp = float('nan')
            error_table.append([col, 'Prophet', round(mae_prophet, 2) if not pd.isna(mae_prophet) else None,
                                round(rmse_prophet, 2) if not pd.isna(rmse_prophet) else None])
            error_table.append([col, 'ARIMA', round(mae_arima, 2) if not pd.isna(mae_arima) else None,
                                round(rmse_arima, 2) if not pd.isna(rmse_arima) else None])
            error_table.append([col, 'ExponentialSmoothing', round(mae_exp, 2) if not pd.isna(mae_exp) else None,
                                round(rmse_exp, 2) if not pd.isna(rmse_exp) else None])
            rmse_candidates = {}
            if not pd.isna(rmse_prophet):
                rmse_candidates['Prophet'] = rmse_prophet
            if not pd.isna(rmse_arima):
                rmse_candidates['ARIMA'] = rmse_arima
            if not pd.isna(rmse_exp):
                rmse_candidates['ExponentialSmoothing'] = rmse_exp
            if rmse_candidates:
                best_model = min(rmse_candidates, key=rmse_candidates.get)
                best_rmse = rmse_candidates[best_model]
            else:
                best_model = None
                best_rmse = None
            if PROPHET_AVAILABLE:
                try:
                    m_full = Prophet(daily_seasonality=True)
                    m_full.fit(ts)
                    future_full = m_full.make_future_dataframe(periods=30)
                    forecast_full = m_full.predict(future_full)
                    df_pred_future_prophet = forecast_full[forecast_full['ds'] > ts['ds'].max()][['ds', 'yhat', 'yhat_lower', 'yhat_upper']].rename(
                        columns={'yhat': 'Predicted', 'yhat_lower': 'Lower', 'yhat_upper': 'Upper'}
                    )
                except Exception:
                    df_pred_future_prophet = pd.DataFrame(columns=['ds', 'Predicted', 'Lower', 'Upper'])
            if STATSMODELS_AVAILABLE:
                try:
                    ts_arima_full = ts.set_index('ds')['y']
                    arima_full = ARIMA(ts_arima_full, order=(1,1,1))
                    arima_full_fit = arima_full.fit()
                    forecast_future_a = arima_full_fit.get_forecast(steps=30)
                    conf_int_future = forecast_future_a.conf_int(alpha=0.05)
                    df_pred_future_arima = pd.DataFrame({
                        'ds': pd.date_range(ts_arima_full.index[-1] + pd.Timedelta(days=1), periods=30),
                        'Predicted': forecast_future_a.predicted_mean.values,
                        'Lower': conf_int_future.iloc[:, 0].values,
                        'Upper': conf_int_future.iloc[:, 1].values
                    })
                except Exception:
                    df_pred_future_arima = pd.DataFrame(columns=['ds', 'Predicted', 'Lower', 'Upper'])
                try:
                    ts_exp_full = ts.set_index('ds')['y']
                    exp_full = ExponentialSmoothing(ts_exp_full, trend='add', seasonal=None, damped_trend=True)
                    exp_full_fit = exp_full.fit()
                    forecast_future_e = exp_full_fit.forecast(steps=30)
                    std_err_full = np.std(exp_full_fit.resid)
                    df_pred_future_exp = pd.DataFrame({
                        'ds': pd.date_range(ts_exp_full.index[-1] + pd.Timedelta(days=1), periods=30),
                        'Predicted': forecast_future_e.values,
                        'Lower': forecast_future_e.values - 1.96 * std_err_full,
                        'Upper': forecast_future_e.values + 1.96 * std_err_full
                    })
                except Exception:
                    df_pred_future_exp = pd.DataFrame(columns=['ds', 'Predicted', 'Lower', 'Upper'])
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=ts['ds'], y=ts['y'], mode='lines+markers', name='Actual', line=dict(color='blue')))
            if not preds_prophet_test.empty:
                fig.add_trace(go.Scatter(x=preds_prophet_test['ds'], y=preds_prophet_test['Predicted'],
                                         mode='lines+markers', name='Prophet (Test)', line=dict(color='green')))
                fig.add_trace(go.Scatter(x=preds_prophet_test['ds'], y=preds_prophet_test['Lower'],
                                         mode='lines', name='Prophet Lower CI', line=dict(color='green', dash='dash')))
                fig.add_trace(go.Scatter(x=preds_prophet_test['ds'], y=preds_prophet_test['Upper'],
                                         mode='lines', name='Prophet Upper CI', line=dict(color='green', dash='dash'), fill='tonexty'))
            if not preds_arima_test.empty:
                fig.add_trace(go.Scatter(x=preds_arima_test['ds'], y=preds_arima_test['Predicted'],
                                         mode='lines+markers', name='ARIMA (Test)', line=dict(color='orange')))
                fig.add_trace(go.Scatter(x=preds_arima_test['ds'], y=preds_arima_test['Lower'],
                                         mode='lines', name='ARIMA Lower CI', line=dict(color='orange', dash='dash')))
                fig.add_trace(go.Scatter(x=preds_arima_test['ds'], y=preds_arima_test['Upper'],
                                         mode='lines', name='ARIMA Upper CI', line=dict(color='orange', dash='dash'), fill='tonexty'))
            if not preds_exp_test.empty:
                fig.add_trace(go.Scatter(x=preds_exp_test['ds'], y=preds_exp_test['Predicted'],
                                         mode='lines+markers', name='Exp Smoothing (Test)', line=dict(color='purple')))
                fig.add_trace(go.Scatter(x=preds_exp_test['ds'], y=preds_exp_test['Lower'],
                                         mode='lines', name='Exp Lower CI', line=dict(color='purple', dash='dash')))
                fig.add_trace(go.Scatter(x=preds_exp_test['ds'], y=preds_exp_test['Upper'],
                                         mode='lines', name='Exp Upper CI', line=dict(color='purple', dash='dash'), fill='tonexty'))
            if best_model == 'Prophet' and not df_pred_future_prophet.empty:
                fig.add_trace(go.Scatter(x=df_pred_future_prophet['ds'], y=df_pred_future_prophet['Predicted'],
                                         mode='lines+markers', name='Best (Future) - Prophet', line=dict(color='red', dash='dash')))
                fig.add_trace(go.Scatter(x=df_pred_future_prophet['ds'], y=df_pred_future_prophet['Lower'],
                                         mode='lines', name='Prophet Future Lower CI', line=dict(color='red', dash='dash')))
                fig.add_trace(go.Scatter(x=df_pred_future_prophet['ds'], y=df_pred_future_prophet['Upper'],
                                         mode='lines', name='Prophet Future Upper CI', line=dict(color='red', dash='dash'), fill='tonexty'))
            elif best_model == 'ARIMA' and not df_pred_future_arima.empty:
                fig.add_trace(go.Scatter(x=df_pred_future_arima['ds'], y=df_pred_future_arima['Predicted'],
                                         mode='lines+markers', name='Best (Future) - ARIMA', line=dict(color='red', dash='dash')))
                fig.add_trace(go.Scatter(x=df_pred_future_arima['ds'], y=df_pred_future_arima['Lower'],
                                         mode='lines', name='ARIMA Future Lower CI', line=dict(color='red', dash='dash')))
                fig.add_trace(go.Scatter(x=df_pred_future_arima['ds'], y=df_pred_future_arima['Upper'],
                                         mode='lines', name='ARIMA Future Upper CI', line=dict(color='red', dash='dash'), fill='tonexty'))
            elif best_model == 'ExponentialSmoothing' and not df_pred_future_exp.empty:
                fig.add_trace(go.Scatter(x=df_pred_future_exp['ds'], y=df_pred_future_exp['Predicted'],
                                         mode='lines+markers', name='Best (Future) - Exp', line=dict(color='red', dash='dash')))
                fig.add_trace(go.Scatter(x=df_pred_future_exp['ds'], y=df_pred_future_exp['Lower'],
                                         mode='lines', name='Exp Future Lower CI', line=dict(color='red', dash='dash')))
                fig.add_trace(go.Scatter(x=df_pred_future_exp['ds'], y=df_pred_future_exp['Upper'],
                                         mode='lines', name='Exp Future Upper CI', line=dict(color='red', dash='dash'), fill='tonexty'))
            fig.update_layout(
                title=f"📈 Actual vs Predicted برای {col} ({unit})",
                xaxis_title="تاریخ",
                yaxis_title=f"مقدار مصرف ({unit})",
                template="plotly_white"
            )
            st.plotly_chart(fig, use_container_width=True, key=f"ml_chart_{col}")
            if best_model is not None:
                st.success(f"💡 بهترین مدل برای {col}: {best_model} (RMSE = {best_rmse:.2f})")
            else:
                st.info(f"⚠️ برای {col} مدل برتر قابل تعیین نیست (خطا یا دادهٔ ناکافی).")
            # خروجی PDF برای Tab9 (برای هر تجهیز)
            if st.button(f"⬇️ دانلود PDF برای {col}", key=f"download_ml_pdf_{col}*v1"):  # 👈 فیکس: unique key
                buffer = io.BytesIO()
                elements = []

                # تولید تصویر با کیفیت بهتر (نیاز به kaleido: pip install kaleido)
                img_buf = io.BytesIO()
                fig.write_image(img_buf, format='png', width=800, height=400, scale=2)
                img_buf.seek(0)
                elements.append(Image(img_buf, width=500, height=300))

                title = f"پیش‌بینی {col} ({unit})"
                if not USE_PERSIAN:
                    title = translations.get("پیش‌بینی", "Forecast") + f" {col}"
                generate_pdf(title, elements, buffer)

                # 👈 دانلود با getvalue() برای اطمینان
                pdf_data = buffer.getvalue()
                st.download_button(
                    label="دانلود PDF",
                    data=pdf_data,
                    file_name=f"tab9*{col}.pdf",
                    mime="application/pdf"
                )

                # چک فونت
                if not USE_PERSIAN:
                    st.warning("⚠️ فونت فارسی پیدا نشد. PDF به انگلیسی تولید شد.")
        if error_table:
            st.markdown("### 📊 جدول خطاها برای همه تجهیزات")
            error_df = pd.DataFrame(error_table, columns=['تجهیز', 'مدل', 'MAE', 'RMSE'])
            st.dataframe(error_df.style.format({'MAE': '{:.2f}', 'RMSE': '{:.2f}'}))
            # خروجی PDF برای جدول خطاها
            if st.button("⬇️ دانلود PDF جدول خطاها Tab9", key="download_tab9_errors_pdf_v1"):  # 👈 فیکس: unique key
                buffer = io.BytesIO()
                elements = []

                data = [error_df.columns.tolist()] + error_df.fillna(0).values.tolist()  # 👈 فیکس: fillna(0)

                # ترجمه هدرها اگر انگلیسی
                translations_local = {
                    "تجهیز": "Equipment",
                    "مدل": "Model",
                    "MAE": "MAE",
                    "RMSE": "RMSE"
                }
                if not USE_PERSIAN and data and isinstance(data[0], list) and len(data[0]) >= 4:
                    data[0][0] = translations_local.get(data[0][0], data[0][0])
                    data[0][1] = translations_local.get(data[0][1], data[0][1])
                    data[0][2] = translations_local.get(data[0][2], data[0][2])
                    data[0][3] = translations_local.get(data[0][3], data[0][3])

                # Reshape متن‌ها برای RTL اگر فارسی (فقط رشته‌ها)
                if USE_PERSIAN:
                    try:
                        for row in data:
                            for i, cell in enumerate(row):
                                if isinstance(cell, str):
                                    row[i] = get_display(arabic_reshaper.reshape(cell))
                    except ImportError:
                        st.warning("برای RTL در جدول، arabic-reshaper و python-bidi رو نصب کن.")

                table = Table(data)
                table.setStyle(TableStyle([
                    ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),  # پس‌زمینه عنوان
                    ('TEXTCOLOR', (0,0), (-1,0), colors.black),  # متن عنوان
                    ('GRID', (0,0), (-1,-1), 0.5, colors.lightgrey),  # 👈 grid کم‌رنگ (نه مشکی)
                    ('ALIGN', (0,0), (-1,-1), 'CENTER')
                ]))
                elements.append(table)

                title = "جدول خطاها"
                if not USE_PERSIAN:
                    title = translations.get(title, title)
                generate_pdf(title, elements, buffer)

                # 👈 دانلود با getvalue() برای اطمینان
                pdf_data = buffer.getvalue()
                st.download_button(
                    label="دانلود PDF",
                    data=pdf_data,
                    file_name="tab9_errors.pdf",
                    mime="application/pdf"
                )

                # چک فونت
                if not USE_PERSIAN:
                    st.warning("⚠️ فونت فارسی پیدا نشد. PDF به انگلیسی تولید شد.")
# ===============================================
# Tab 10: تحلیل دیتا (با فیکس داده کم از شیت‌های مختلف + گزینه‌های نمایش + تست Normality)
# ===============================================
with tabs[9]:
    st.subheader("🔬 تحلیل دیتا و تعیین متغیرهای تاثیرگذار")
    target_var = st.selectbox(
        "📌 انتخاب متغیر وابسته:",
        selected_equipment,
        key="analysis_target_select"
    )
    predictor_vars = st.multiselect(
        "📌 انتخاب متغیرهای مستقل:",
        [col for col in selected_equipment if col != target_var],
        key="analysis_predictors_multiselect"
    )

    # 👈 فیکس: فیلتر شیت اختیاری (پیش‌فرض: همه شیت‌ها برای corr)
    use_sheet_filter = st.checkbox("🔄 فیلتر بر اساس شیت خاص (برای OLS دقیق‌تر)", value=False, key="use_sheet_filter")

    if 'کارخانه' in filtered_df.columns and use_sheet_filter:
        selected_sheet = st.selectbox(
            "📂 انتخاب شیت (کارخانه) برای تحلیل:",
            filtered_df['کارخانه'].unique(),
            key="sheet_filter_select"
        )
        df_sheet = filtered_df[filtered_df['کارخانه'] == selected_sheet].copy()
        st.info(f"📊 داده‌های شیت '{selected_sheet}': {len(df_sheet)} رکورد")
    else:
        df_sheet = filtered_df.copy()
        selected_sheet = "همه شیت‌ها"
        st.info(f"📊 داده‌های کلی: {len(df_sheet)} رکورد (از همه شیت‌ها)")

    # 👈 جدید: نمایش تعداد داده‌های موجود برای هر متغیر (برای دیباگ)
    if target_var and predictor_vars:
        st.markdown("### 📈 وضعیت داده‌ها (تعداد non-NaN)")
        data_status = []
        for col in [target_var] + predictor_vars:
            non_nan_count = df_sheet[col].notna().sum()
            data_status.append({"متغیر": col, "تعداد داده معتبر": non_nan_count})
        status_df = pd.DataFrame(data_status)
        st.dataframe(status_df.style.background_gradient(cmap="Greens", subset=["تعداد داده معتبر"]))
        if any(status_df["تعداد داده معتبر"] < 2):
            st.warning("⚠️ برخی متغیرها داده کم دارن – همبستگی pairwise محاسبه می‌شه.")

    if target_var and predictor_vars:
        target_unit = get_unit_for_column(filtered_df, target_var, st.session_state.custom_units)  # 👈 تغییر: واحد target
        df_selected = df_sheet[[target_var] + predictor_vars].copy()

        # 👈 فیکس: fillna(0) برای corr (جلوگیری از dropna کامل)، اما corr با min_periods=2
        df_for_corr = df_selected.fillna(0)
        if len(df_for_corr) < 2:
            st.warning("⚠️ داده کافی برای تحلیل وجود ندارد (حداقل ۲ رکورد لازم است). شیت دیگه امتحان کن یا فیلتر شیت رو خاموش کن.")
        else:
            st.markdown("### 📊 ماتریس همبستگی")
            # 👈 فیکس: corr با min_periods=2 (pairwise، بدون warning برای داده کم)
            corr = df_for_corr.corr(method='pearson', min_periods=2)
            fig_corr = px.imshow(
                corr,
                text_auto=True,
                color_continuous_scale="RdBu_r",
                zmin=-1, zmax=1,
                title=f"ماتریس همبستگی ({target_unit}) - شیت: {selected_sheet}"
            )
            st.plotly_chart(fig_corr, use_container_width=True)
            st.markdown("**💡 تفسیر همبستگی:**")
            for col in predictor_vars:
                r = corr.loc[target_var, col]
                if abs(r) > 0.7:
                    strength = "قوی"
                elif abs(r) > 0.5:
                    strength = "قابل توجه"
                else:
                    strength = "ضعیف"
                st.write(f"{col}: همبستگی با {target_var} = {r:.2f} → {strength}")

            st.markdown("### 📈 تحلیل رگرسیون")
            st.markdown("#### رگرسیون تک‌متغیره")
            single_results = []
            # 👈 فیکس: dropna فقط برای OLS (داده‌های مشترک)
            df_for_ols = df_selected.dropna()
            if len(df_for_ols) < 10:
                st.warning(f"⚠️ داده کم در شیت '{selected_sheet}' (<10 row مشترک). corr کلی محاسبه شد، اما OLS تقریبیه. فیلتر شیت رو خاموش کن.")
            for col in predictor_vars:
                # 👈 فیکس: چک داده مشترک برای هر col (حداقل ۲)
                temp_df = df_for_ols[[target_var, col]].dropna()
                if len(temp_df) < 2:
                    st.warning(f"⚠️ داده مشترک برای {col} و {target_var} کافی نیست (حداقل ۲ رکورد).")
                    continue
                X = temp_df[[col]]
                X_const = sm.add_constant(X)
                y_temp = temp_df[target_var]
                try:
                    model = sm.OLS(y_temp, X_const).fit()
                    r2 = model.rsquared
                    p_val = model.pvalues.get(col, np.nan)
                    significant = (not pd.isna(p_val)) and p_val < 0.05 and r2 > 0.75
                    single_results.append({
                        "Variable": col,
                        "R²": r2,
                        "p-value": p_val,
                        "Impactful": "✅" if significant else "❌"
                    })
                except Exception as e:
                    st.warning(f"⚠️ خطا در رگرسیون {col}: {e}")
                    continue

            if single_results:
                st.table(pd.DataFrame(single_results))
            else:
                st.info("هیچ رگرسیون موفقی اجرا نشد. داده‌های مشترک رو چک کن.")

            multi_summary = None  # 👈 Fix: initialize برای چک locals
            if len(predictor_vars) > 1:
                st.markdown("#### رگرسیون چندمتغیره")
                # 👈 فیکس: چک داده مشترک برای همه predictorها
                temp_multi = df_selected[predictor_vars + [target_var]].dropna()
                if len(temp_multi) < 2:
                    st.warning("⚠️ داده برای رگرسیون چندمتغیره کافی نیست (داده مشترک کم).")
                else:
                    X_multi = temp_multi[predictor_vars]
                    X_multi_const = sm.add_constant(X_multi)
                    y_multi = temp_multi[target_var]
                    if len(y_multi) < 2 or X_multi_const.shape[0] == 0:
                        st.warning("⚠️ X_multi_const خالی.")
                    else:
                        try:
                            multi_model = sm.OLS(y_multi, X_multi_const).fit()
                            multi_summary = pd.DataFrame({
                                "Variable": multi_model.params.index[1:],
                                "Coefficient": multi_model.params.values[1:],
                                "p-value": multi_model.pvalues.values[1:],
                                "Significant": ["✅" if p < 0.05 else "❌" for p in multi_model.pvalues.values[1:]]
                            })
                            st.table(multi_summary)
                        except Exception as e:
                            st.warning(f"⚠️ خطا در رگرسیون چندمتغیره: {e}")

            # 👈 جدید: سه checkbox برای گزینه‌های نمایش (پیش‌فرض همه تیک‌خورده) + checkbox تست normality
            st.markdown("### ⚙️ گزینه‌های نمایش نمودار")
            col1, col2, col3 = st.columns(3)
            with col1:
                show_iqr = st.checkbox("IQR Limits (Tukey)", value=True, key="show_iqr")
            with col2:
                show_shewhart = st.checkbox("Shewhart Limits (3σ)", value=True, key="show_shewhart")
            with col3:
                show_simple = st.checkbox("Box Plot ساده (بدون Limits)", value=True, key="show_simple")
            test_normality = st.checkbox("🔍 تست نرمال بودن داده‌ها (Shapiro-Wilk)", value=False, key="test_normality")

            # 👈 جدید: اجرای تست normality اگر تیک‌خورده (برای هر col)
            if test_normality:
                st.markdown("### 📊 نتایج تست نرمال بودن (Shapiro-Wilk)")
                normality_results = []
                from scipy import stats  # 👈 import scipy برای shapiro
                for col in predictor_vars + [target_var]:
                    col_data = df_sheet[col].dropna().clip(lower=0)  # داده معتبر + clip
                    if len(col_data) < 3:  # حداقل ۳ برای shapiro
                        normality_results.append({"متغیر": col, "p-value": "N/A", "نتیجه": "داده کم"})
                        continue
                    stat, p_val = stats.shapiro(col_data)
                    result = "نرمال ✅" if p_val > 0.05 else "غیرنرمال ⚠️"
                    normality_results.append({"متغیر": col, "p-value": round(p_val, 4), "نتیجه": result})
                norm_df = pd.DataFrame(normality_results)
                st.table(norm_df)
                if norm_df["نتیجه"].str.contains("غیرنرمال").any():
                    st.warning("⚠️ برخی داده‌ها غیرنرمالن – Shewhart ممکنه دقیق نباشه (از IQR استفاده کن).")
                else:
                    st.success("✅ همه داده‌ها نرمالن – Shewhart مناسبه!")

            # 👈 فیکس: نمایش نمودارها بر اساس checkboxها
            st.markdown("### 📦 نمودارهای Box Plot")
            for col in predictor_vars + [target_var]:
                col_unit = get_unit_for_column(filtered_df, col, st.session_state.custom_units)  # 👈 تغییر: واحد هر col
                # 👈 فیکس: آماده‌سازی داده (تاریخ + col + clip منفی‌ها به ۰)
                box_data = df_sheet[["تاریخ", col]].dropna(subset=[col]).copy()
                box_data[col] = box_data[col].clip(lower=0)  # 👈 جدید: منفی‌ها رو ۰ کن
                if len(box_data) < 4:  # حداقل ۴ برای محاسبات
                    st.warning(f"⚠️ داده برای {col} کمه (<۴). Skip.")
                    continue
                # 👈 فیکس: فرمت تاریخ برای نمایش بهتر در x (Jalali)
                box_data["تاریخ"] = box_data["تاریخ"].apply(safe_jalali_format)

                # 👈 شرطی: نمایش بر اساس checkbox
                if show_simple:
                    # Box Plot ساده
                    fig_simple = px.box(
                        box_data,
                        x="تاریخ",
                        y=col,
                        points="all",
                        title=f"Box Plot ساده: {col} ({col_unit})",
                        orientation="v"
                    )
                    fig_simple.update_layout(xaxis_tickangle=-45, height=400)
                    st.plotly_chart(fig_simple, use_container_width=True)

                if show_iqr:
                    # IQR Limits
                    Q1 = box_data[col].quantile(0.25)
                    Q3 = box_data[col].quantile(0.75)
                    IQR = Q3 - Q1
                    LCL_iqr = max(0, Q1 - 1.5 * IQR)
                    UCL_iqr = Q3 + 1.5 * IQR
                    fig_iqr = px.box(
                        box_data,
                        x="تاریخ",
                        y=col,
                        points="all",
                        title=f"IQR Limits: {col} ({col_unit})",
                        orientation="v"
                    )
                    fig_iqr.add_hline(y=UCL_iqr, line_dash="dash", line_color="red", annotation_text=f"UCL = {UCL_iqr:.2f}")
                    fig_iqr.add_hline(y=LCL_iqr, line_dash="dash", line_color="green", annotation_text=f"LCL = {LCL_iqr:.2f}")
                    fig_iqr.update_layout(xaxis_tickangle=-45, height=400)
                    st.plotly_chart(fig_iqr, use_container_width=True)

                if show_shewhart:
                    # Shewhart Limits
                    CL = box_data[col].mean()
                    sigma = box_data[col].std()
                    UCL_shewhart = CL + 3 * sigma
                    LCL_raw = CL - 3 * sigma
                    LCL_shewhart = max(0, LCL_raw)
                    fig_shewhart = px.box(
                        box_data,
                        x="تاریخ",
                        y=col,
                        points="all",
                        title=f"Shewhart Limits: {col} ({col_unit})",
                        orientation="v"
                    )
                    fig_shewhart.add_hline(y=CL, line_dash="solid", line_color="black", annotation_text=f"CL = {CL:.2f}")
                    fig_shewhart.add_hline(y=UCL_shewhart, line_dash="dash", line_color="red", annotation_text=f"UCL = {UCL_shewhart:.2f}")
                    fig_shewhart.add_hline(y=LCL_shewhart, line_dash="dash", line_color="green", annotation_text=f"LCL = {LCL_shewhart:.2f}")
                    fig_shewhart.update_layout(xaxis_tickangle=-45, height=400)
                    st.plotly_chart(fig_shewhart, use_container_width=True)

                # 👈 تحلیل کلی (برای هر col، بر اساس گزینه فعال)
                # از اجتماع ایندکس‌ها استفاده می‌شود تا نقاطی که هم‌زمان توسط IQR و هم Shewhart
                # به‌عنوان پرت شناسایی می‌شوند، دوبار شمرده نشوند.
                outlier_indices = set()
                if show_iqr:
                    outlier_indices |= set(box_data[box_data[col] > UCL_iqr].index)
                if show_shewhart:
                    outlier_indices |= set(box_data[box_data[col] > UCL_shewhart].index)
                outliers_count = len(outlier_indices)
                if outliers_count > 0:
                    st.warning(f"⚠️ {outliers_count} outlier در {col} شناسایی شد!")
                else:
                    st.success(f"✅ {col} پایداره – هیچ outlier!")

            # خروجی PDF (بدون تغییر، اما با فیکس locals)
            if st.button("⬇️ دانلود PDF Tab10", key="download_tab10_pdf_v1"):
                buffer = io.BytesIO()
                elements = []

                # تولید تصویر با کیفیت بهتر (نیاز به kaleido: pip install kaleido)
                img_buf_corr = io.BytesIO()
                fig_corr.write_image(img_buf_corr, format='png', width=800, height=400, scale=2)
                img_buf_corr.seek(0)
                elements.append(Image(img_buf_corr, width=500, height=300))

                # جدول‌های رگرسیون
                if single_results:
                    single_df = pd.DataFrame(single_results)
                    data_single = [single_df.columns.tolist()] + single_df.fillna(0).values.tolist()  # 👈 فیکس: fillna(0)

                    # ترجمه هدرها اگر انگلیسی
                    translations_local = {
                        "Variable": "Variable",
                        "R²": "R²",
                        "p-value": "p-value",
                        "Impactful": "Impactful"
                    }
                    if not USE_PERSIAN and data_single and isinstance(data_single[0], list):
                        for i, header in enumerate(data_single[0]):
                            data_single[0][i] = translations_local.get(header, header)

                    # Reshape متن‌ها برای RTL اگر فارسی (فقط رشته‌ها)
                    if USE_PERSIAN and RTL_AVAILABLE:
                        try:
                            for row in data_single:
                                for i, cell in enumerate(row):
                                    if isinstance(cell, str):
                                        row[i] = get_display(arabic_reshaper.reshape(cell))
                        except ImportError:
                            st.warning("برای RTL در جدول، arabic-reshaper و python-bidi رو نصب کن.")

                    table_single = Table(data_single)
                    table_single.setStyle(TableStyle([
                        ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),  # پس‌زمینه عنوان
                        ('TEXTCOLOR', (0,0), (-1,0), colors.black),  # متن عنوان
                        ('GRID', (0,0), (-1,-1), 0.5, colors.lightgrey),  # 👈 grid کم‌رنگ (نه مشکی)
                        ('ALIGN', (0,0), (-1,-1), 'CENTER')
                    ]))
                    elements.append(table_single)

                # 👈 Fix: چک multi_summary با if
                if 'multi_summary' in locals() and multi_summary is not None:
                    data_multi = [multi_summary.columns.tolist()] + multi_summary.fillna(0).values.tolist()  # 👈 فیکس: fillna(0)

                    # ترجمه هدرها اگر انگلیسی
                    translations_local = {
                        "Variable": "Variable",
                        "Coefficient": "Coefficient",
                        "p-value": "p-value",
                        "Significant": "Significant"
                    }
                    if not USE_PERSIAN and data_multi and isinstance(data_multi[0], list):
                        for i, header in enumerate(data_multi[0]):
                            data_multi[0][i] = translations_local.get(header, header)

                    # Reshape متن‌ها برای RTL اگر فارسی (فقط رشته‌ها)
                    if USE_PERSIAN and RTL_AVAILABLE:
                        try:
                            for row in data_multi:
                                for i, cell in enumerate(row):
                                    if isinstance(cell, str):
                                        row[i] = get_display(arabic_reshaper.reshape(cell))
                        except ImportError:
                            st.warning("برای RTL در جدول، arabic_reshaper و python-bidi رو نصب کن.")

                    table_multi = Table(data_multi)
                    table_multi.setStyle(TableStyle([
                        ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),  # پس‌زمینه عنوان
                        ('TEXTCOLOR', (0,0), (-1,0), colors.black),  # متن عنوان
                        ('GRID', (0,0), (-1,-1), 0.5, colors.lightgrey),  # 👈 grid کم‌رنگ (نه مشکی)
                        ('ALIGN', (0,0), (-1,-1), 'CENTER')
                    ]))
                    elements.append(table_multi)

                # 👈 Box Plots در PDF بر اساس گزینه‌های فعال — هر گزینهٔ فعال، نمودار مستقل خودش را دارد
                # (قبلاً از if/elif/else استفاده می‌شد که فقط یکی از سه نوع را در PDF می‌گذاشت،
                # حتی اگر کاربر هر سه گزینه را روی صفحه فعال کرده باشد.)
                for col in predictor_vars + [target_var]:
                    col_unit = get_unit_for_column(filtered_df, col, st.session_state.custom_units)
                    box_data_pdf = df_sheet[["تاریخ", col]].dropna(subset=[col]).copy()
                    box_data_pdf[col] = box_data_pdf[col].clip(lower=0)
                    box_data_pdf["تاریخ"] = box_data_pdf["تاریخ"].apply(safe_jalali_format)
                    if show_simple:
                        fig_box_simple = px.box(box_data_pdf, x="تاریخ", y=col, points="all", title=f"ساده: {col}", orientation="v")
                        fig_box_simple.update_layout(xaxis_tickangle=-45)
                        img_buf_box = io.BytesIO()
                        fig_box_simple.write_image(img_buf_box, format='png', width=800, height=400, scale=2)
                        img_buf_box.seek(0)
                        elements.append(Image(img_buf_box, width=500, height=300))
                    if show_iqr:
                        Q1_pdf = box_data_pdf[col].quantile(0.25)
                        Q3_pdf = box_data_pdf[col].quantile(0.75)
                        IQR_pdf = Q3_pdf - Q1_pdf
                        LCL_iqr_pdf = max(0, Q1_pdf - 1.5 * IQR_pdf)
                        UCL_iqr_pdf = Q3_pdf + 1.5 * IQR_pdf
                        fig_box_iqr = px.box(box_data_pdf, x="تاریخ", y=col, points="all", title=f"IQR: {col}", orientation="v")
                        fig_box_iqr.add_hline(y=UCL_iqr_pdf, line_dash="dash", line_color="red", annotation_text=f"UCL={UCL_iqr_pdf:.2f}")
                        fig_box_iqr.add_hline(y=LCL_iqr_pdf, line_dash="dash", line_color="green", annotation_text=f"LCL={LCL_iqr_pdf:.2f}")
                        fig_box_iqr.update_layout(xaxis_tickangle=-45)
                        img_buf_box = io.BytesIO()
                        fig_box_iqr.write_image(img_buf_box, format='png', width=800, height=400, scale=2)
                        img_buf_box.seek(0)
                        elements.append(Image(img_buf_box, width=500, height=300))
                    if show_shewhart:
                        CL_pdf = box_data_pdf[col].mean()
                        sigma_pdf = box_data_pdf[col].std()
                        UCL_pdf = CL_pdf + 3 * sigma_pdf
                        LCL_pdf = max(0, CL_pdf - 3 * sigma_pdf)
                        fig_box_shewhart = px.box(box_data_pdf, x="تاریخ", y=col, points="all", title=f"Shewhart: {col}", orientation="v")
                        fig_box_shewhart.add_hline(y=CL_pdf, line_dash="solid", line_color="black", annotation_text=f"CL={CL_pdf:.2f}")
                        fig_box_shewhart.add_hline(y=UCL_pdf, line_dash="dash", line_color="red", annotation_text=f"UCL={UCL_pdf:.2f}")
                        fig_box_shewhart.add_hline(y=LCL_pdf, line_dash="dash", line_color="green", annotation_text=f"LCL={LCL_pdf:.2f}")
                        fig_box_shewhart.update_layout(xaxis_tickangle=-45)
                        img_buf_box = io.BytesIO()
                        fig_box_shewhart.write_image(img_buf_box, format='png', width=800, height=400, scale=2)
                        img_buf_box.seek(0)
                        elements.append(Image(img_buf_box, width=500, height=300))

                title = f"تحلیل دیتا ({target_unit})"
                if not USE_PERSIAN:
                    title = translations.get(title, title)
                generate_pdf(title, elements, buffer)

                # 👈 دانلود با getvalue() برای اطمینان
                pdf_data = buffer.getvalue()
                st.download_button(
                    label="دانلود PDF",
                    data=pdf_data,
                    file_name=f"تحلیل_دیتا_{target_var}_{safe_jalali_format(pd.Timestamp.today())}.pdf",
                    mime="application/pdf"
                )

                # چک فونت 👈 Fix: available_fonts به USE_PERSIAN
                if not USE_PERSIAN:
                    st.warning("⚠️ فونت فارسی پیدا نشد. PDF به انگلیسی تولید شد.")
    else:
        st.info("⚠️ لطفاً متغیر وابسته و حداقل یک متغیر مستقل انتخاب کنید تا تحلیل شروع شود.")
# ===============================================
# Tab 11: ناهنجاری‌ها
# ===============================================
# ----------- Tab11: تشخیص ناهنجاری‌ها -----------
with tabs[10]:
    st.subheader("🚨 تشخیص ناهنجاری‌ها و هشدارها")
    anomaly_col = st.selectbox("🔌 انتخاب تجهیز برای تحلیل ناهنجاری:",selected_equipment, key="anomaly_select")

    if anomaly_col:
        unit = get_unit_for_column(filtered_df, anomaly_col, st.session_state.custom_units)  # 👈 تغییر: واحد تجهیز
        
        # 👈 بهبود: فیلدهای درصد تغییر با moving average
        col1, col2 = st.columns(2)
        with col1:
            change_threshold = st.number_input(
                "درصد تغییر مجاز برای تشخیص ناهنجاری (%):", 
                min_value=0.0, 
                value=10.0,  # 👈 پیش‌فرض 10%
                step=1.0, 
                help="هر تغییری بیش از این درصد، به عنوان ناهنجاری علامت‌گذاری می‌شود.",
                key="change_threshold"
            )
        with col2:
            window_size = st.number_input(
                "طول میانگین متحرک (روز):", 
                min_value=1, 
                value=3,  # 👈 جدید: پیش‌فرض 3 روز برای روند بهتر
                step=1, 
                help="تغییر نسبت به میانگین این تعداد روز قبل محاسبه می‌شود (1=رکورد قبلی).",
                key="window_size"
            )
        change_type = st.selectbox(
            "نوع تغییر:", 
            ["مطلق (بالا/پایین)", "فقط افزایش", "فقط کاهش"], 
            index=0,  # 👈 جدید: گزینه برای جهت
            key="change_type"
        )

        df_anomaly = filtered_df[["تاریخ", "تاریخ شمسی", anomaly_col]].dropna().copy()
        if len(df_anomaly) < window_size + 1:  # 👈 بهبود: حداقل داده برای window
            st.warning(f"⚠️ داده کافی برای تحلیل ناهنجاری در {anomaly_col} وجود ندارد (حداقل {window_size+1} رکورد لازم است).")
        else:
            # 👈 مرتب‌سازی بر اساس تاریخ
            df_anomaly = df_anomaly.sort_values("تاریخ").reset_index(drop=True)
            
            # 👈 بهبود: محاسبه moving average و درصد تغییر نسبت به آن
            df_anomaly[f"{anomaly_col}_ma"] = df_anomaly[anomaly_col].rolling(window=window_size, min_periods=1).mean()
            df_anomaly["percent_change_ma"] = ((df_anomaly[anomaly_col] - df_anomaly[f"{anomaly_col}_ma"].shift(1)) / df_anomaly[f"{anomaly_col}_ma"].shift(1)) * 100  # تغییر نسبت به MA قبلی
            
            # 👈 فیلتر صفرها برای IQR/IsolationForest (مثل قبل)
            non_zero_data = df_anomaly[df_anomaly[anomaly_col] > 0]
            if len(non_zero_data) < 2:
                st.warning(f"⚠️ داده غیرصفر کافی برای تحلیل ناهنجاری در {anomaly_col} وجود ندارد.")
            else:
                X = df_anomaly[[anomaly_col]].values
                iso_forest = IsolationForest(contamination=0.1, random_state=42)
                df_anomaly["is_anomaly"] = iso_forest.fit_predict(X)
                df_anomaly["is_anomaly"] = df_anomaly["is_anomaly"] == -1
                
                Q1 = non_zero_data[anomaly_col].quantile(0.25)
                Q3 = non_zero_data[anomaly_col].quantile(0.75)
                IQR = Q3 - Q1
                LCL = max(0, Q1 - 1.5 * IQR)
                UCL = Q3 + 1.5 * IQR
                
                df_anomaly.loc[df_anomaly[anomaly_col] == 0, "is_anomaly"] = False
                
                # 👈 بهبود: تشخیص بر اساس نوع تغییر
                pct_change = df_anomaly["percent_change_ma"]
                if change_type == "فقط افزایش":
                    is_significant_change = pct_change > change_threshold
                elif change_type == "فقط کاهش":
                    is_significant_change = pct_change < -change_threshold
                else:  # مطلق
                    is_significant_change = abs(pct_change) > change_threshold
                
                df_anomaly["is_anomaly"] = df_anomaly["is_anomaly"] | is_significant_change
                
                # 👈 آمار
                st.info(f"💡 آمار تغییرات (MA {window_size}-روزه): میانگین {abs(pct_change).mean():.1f}% | بیشینه {abs(pct_change).max():.1f}% | ناهنجاری‌های تغییر: {is_significant_change.sum()} مورد")
            
                # 👈 نمودار با خط MA
                fig_anomaly = go.Figure()
                normal_data = df_anomaly[~df_anomaly["is_anomaly"]]
                fig_anomaly.add_trace(go.Scatter(
                    x=normal_data["تاریخ"], y=normal_data[anomaly_col], mode="markers", name="نرمال",
                    marker=dict(color="blue", size=8)
                ))
                anomaly_data = df_anomaly[df_anomaly["is_anomaly"]]
                if not anomaly_data.empty:
                    change_anomalies = anomaly_data[is_significant_change]
                    other_anomalies = anomaly_data[~is_significant_change]
                
                    if not other_anomalies.empty:
                        fig_anomaly.add_trace(go.Scatter(
                            x=other_anomalies["تاریخ"], y=other_anomalies[anomaly_col], mode="markers", name="ناهنجاری (IQR/Isolation)",
                            marker=dict(color="red", size=12, symbol="x")
                        ))
                
                    if not change_anomalies.empty:
                        fig_anomaly.add_trace(go.Scatter(
                            x=change_anomalies["تاریخ"], y=change_anomalies[anomaly_col], mode="markers",
                            name=f"ناهنجاری (تغییر {change_type.lower()} >{change_threshold}%)",
                            marker=dict(color="orange", size=12, symbol="diamond")
                        ))
            
                # 👈 جدید: خط moving average
                fig_anomaly.add_trace(go.Scatter(
                    x=df_anomaly["تاریخ"], y=df_anomaly[f"{anomaly_col}_ma"], mode="lines", name=f"میانگین متحرک ({window_size}-روزه)",
                    line=dict(color="green", dash="dot")
                ))
            
                fig_anomaly.add_hline(y=UCL, line_dash="dash", line_color="red", annotation_text=f"UCL = {UCL:.2f} {unit}")
                fig_anomaly.add_hline(y=LCL, line_dash="dash", line_color="green", annotation_text=f"LCL = {LCL:.2f} {unit}")
                fig_anomaly.update_layout(
                    title=f"📊 ناهنجاری‌ها در {anomaly_col} ({unit}) - Threshold: {change_threshold}% ({change_type}, MA{window_size})",
                    xaxis_title="تاریخ", yaxis_title=f"مصرف ({unit})", template="plotly_white", height=500
                )
                st.plotly_chart(fig_anomaly, use_container_width=True)
            
                # 👈 بقیه کد مثل قبل (جدول، تحلیل، PDF) – فقط ستون percent_change_ma رو جایگزین کن
                if not anomaly_data.empty:
                    st.warning(f"⚠️ {len(anomaly_data)} ناهنجاری! (شامل {is_significant_change.sum()} تغییر {change_type.lower()})")
                    st.markdown("📋 **جدول ناهنجاری‌ها**")
                    anomaly_table = anomaly_data[["تاریخ شمسی", anomaly_col, "percent_change_ma"]].rename(
                        columns={anomaly_col: f"مصرف ({unit})", "تاریخ شمسی": "تاریخ", "percent_change_ma": "درصد تغییر MA (%)"}
                    )
                    anomaly_table[f"مصرف ({unit})"] = anomaly_table[f"مصرف ({unit})"].round(2)
                    anomaly_table["درصد تغییر MA (%)"] = anomaly_table["درصد تغییر MA (%)"].round(1)
                    st.dataframe(anomaly_table, use_container_width=True)
                    st.markdown("### 🔎 تحلیل")
                    for idx, row in anomaly_data.iterrows():
                        date_sh = row["تاریخ شمسی"]
                        value = row[anomaly_col]
                        pct_change = row["percent_change_ma"]
                        if abs(pct_change) > change_threshold or (change_type != "مطلق" and pct_change * (-1 if change_type == "فقط کاهش" else 1) > change_threshold):
                            reason = f"تغییر {change_type} {pct_change:+.1f}% (>{change_threshold}%) – شوک روند"
                        elif value > UCL:
                            reason = "بالای UCL (fault/over)"
                        elif value < LCL:
                            reason = "پایین LCL (مشکل فنی)"
                        else:
                            reason = "IsolationForest"
                        st.write(f"- {date_sh} | {value:.2f} {unit} | تغییر: {pct_change:+.1f}% | {reason}")
                else:
                    st.success("✅ هیچ ناهنجاری (صفرها طبیعی).")
            
                # دانلود PDF گزارش ناهنجاری‌ها
                if st.button("⬇️ دانلود گزارش ناهنجاری‌ها (PDF)", key="download_tab11_pdf"):
                    buffer = io.BytesIO()
                    elements = []
                    header = f"""
                    <b>گزارش تشخیص ناهنجاری — {anomaly_col}</b><br/>
                    آستانهٔ تغییر: {change_threshold}% ({change_type}) | میانگین متحرک: {window_size} روزه<br/>
                    تعداد ناهنجاری: {len(anomaly_data)} از {len(df_anomaly)} رکورد<br/>
                    تاریخ گزارش: {safe_jalali_format(pd.Timestamp.today())}
                    """
                    elements.append(Paragraph(rtl(header), ParagraphStyle('Title', fontName=FONT_NAME, fontSize=13, alignment=1, spaceAfter=25)))
                    # تصویر نمودار
                    try:
                        img_buf = io.BytesIO()
                        fig_anomaly.write_image(img_buf, format="png", engine="kaleido", width=1100, height=600, scale=2.5)
                        img_buf.seek(0)
                        elements.append(Image(img_buf, width=520, height=290))
                        elements.append(Spacer(1, 15))
                    except Exception as e:
                        st.warning(f"نمودار در PDF اضافه نشد: {e}")
                    # جدول ناهنجاری‌ها
                    if not anomaly_data.empty:
                        table_data = [["تاریخ", f"مصرف ({unit})", "درصد تغییر MA (%)"]]
                        for _, row in anomaly_table.iterrows():
                            table_data.append([
                                rtl(str(row["تاریخ"])),
                                rtl(f"{row[f'مصرف ({unit})']:,.2f}"),
                                rtl(f"{row['درصد تغییر MA (%)']:+.1f}")
                            ])
                        table = Table(table_data, repeatRows=1)
                        table.setStyle(TableStyle([
                            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#C74A1B")),
                            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                            ('FONTNAME', (0, 0), (-1, -1), FONT_NAME),
                            ('FONTSIZE', (0, 0), (-1, -1), 9),
                            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                        ]))
                        elements.append(table)
                    else:
                        elements.append(Paragraph(rtl("هیچ ناهنجاری‌ای شناسایی نشد."), ParagraphStyle('Normal', fontName=FONT_NAME, alignment=1)))
                    generate_pdf(f"گزارش ناهنجاری — {anomaly_col}", elements, buffer)
                    buffer.seek(0)
                    st.download_button(
                        "دانلود گزارش ناهنجاری‌ها (PDF)",
                        buffer.getvalue(),
                        f"ناهنجاری_{anomaly_col}_{safe_jalali_format(pd.Timestamp.today())}.pdf",
                        "application/pdf",
                        key="download_tab11_pdf_final"
                    )
# ===============================================
# Tab 12: گزارش زیست‌محیطی و پایداری (ISO 14001 & 50001)
# ===============================================
with tabs[11]:
    st.subheader("# گزارش زیست‌محیطی و پایداری — ردپای کربن")
    if not selected_equipment:
        st.info("لطفاً از سایدبار حداقل یک تجهیز انتخاب کنید.")
    else:
        # فقط تجهیزاتی که واحدشان انرژی الکتریکی است (MWh/kWh) برای محاسبه CO₂ معنادارند.
        # واحدهایی مثل تن، تومان، % یا m³ نباید مستقیماً به‌عنوان kWh در فرمول ضرب شوند.
        energy_equipment = [
            col for col in selected_equipment
            if get_unit_for_column(filtered_df, col, st.session_state.custom_units) in ("MWh", "kWh")
        ]
        non_energy_equipment = [col for col in selected_equipment if col not in energy_equipment]
        if non_energy_equipment:
            st.caption(
                "⚠️ تجهیزات با واحد غیر MWh/kWh (مثل تن، تومان، %، m³) از فهرست انتخاب زیر حذف شده‌اند، "
                "چون فرمول انتشار CO₂ این تب فقط برای مصرف برق معتبر است: "
                + "، ".join(non_energy_equipment)
            )
        if not energy_equipment:
            st.warning("هیچ‌کدام از تجهیزات انتخاب‌شده در سایدبار واحد MWh/kWh ندارند — محاسبه انتشار CO₂ برای آن‌ها معنادار نیست.")
        else:
            env_cols = st.multiselect(
                "انتخاب تجهیزات برای محاسبه انتشار CO₂ (فقط تجهیزات برقی):",
                options=energy_equipment,
                default=energy_equipment[:3] if len(energy_equipment) >= 3 else energy_equipment,
                key="env_multiselect"
            )
            if not env_cols:
                st.warning("حداقل یک تجهیز انتخاب کنید.")
            else:
                # تنظیمات پیشرفته
                col1, col2 = st.columns(2)
                with col1:
                    co2_factor = st.number_input(
                        "فاکتور انتشار CO₂ (کیلوگرم CO₂ به ازای هر kWh)",
                        min_value=0.0, max_value=2.0, value=0.495, step=0.005,
                        help="ایران: ≈0.495 kg CO₂/kWh (میانگین شبکه برق)",
                        key="co2_factor"
                    )
                with col2:
                    reduction_target = st.slider(
                        "هدف کاهش انتشار CO₂ نسبت به وضعیت فعلی (%)",
                        min_value=0, max_value=100, value=20, step=5,
                        key="reduction_target"
                    )
                # محاسبه انتشار CO₂
                df_env = filtered_df[["تاریخ", "تاریخ شمسی"] + env_cols].copy()
                for col in env_cols:
                    unit = get_unit_for_column(filtered_df, col, st.session_state.custom_units)
                    # تبدیل به kWh (فقط MWh یا kWh مجاز است، طبق فیلتر بالا)
                    kwh = df_env[col] * 1000 if unit == "MWh" else df_env[col]
                    df_env[f"CO2_{col}"] = kwh * co2_factor
                co2_cols = [f"CO2_{col}" for col in env_cols]
                df_env["CO2_Total"] = df_env[co2_cols].sum(axis=1)
                # نمودار روند انتشار
                fig_trend = px.area(
                    df_env, x="تاریخ", y="CO2_Total",
                    title="روند انتشار دی‌اکسید کربن (CO₂) در طول زمان",
                    color_discrete_sequence=["#C74A1B"],
                    template="plotly_white"
                )
                fig_trend.update_layout(
                    xaxis_title="تاریخ",
                    yaxis_title="انتشار CO₂ (کیلوگرم)",
                    height=500,
                    hovermode="x unified"
                )
                st.plotly_chart(fig_trend, use_container_width=True)
                # نمودار دایره‌ای سهم تجهیزات
                total_per_equip = df_env[co2_cols].sum()
                pie_data = pd.DataFrame({
                    "تجهیز": [col for col in env_cols],
                    "انتشار CO₂ (kg)": [total_per_equip[f"CO2_{col}"] for col in env_cols]
                })
                total_co2_pie = pie_data["انتشار CO₂ (kg)"].sum()
                if total_co2_pie > 0:
                    pie_data["سهم (%)"] = (pie_data["انتشار CO₂ (kg)"] / total_co2_pie * 100).round(1)
                    fig_pie = px.pie(
                        pie_data,
                        names="تجهیز",
                        values="انتشار CO₂ (kg)",
                        title="سهم هر تجهیز در انتشار کل CO₂",
                        color_discrete_sequence=px.colors.sequential.Oranges_r,
                        hole=0.4
                    )
                    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                    st.plotly_chart(fig_pie, use_container_width=True)
                else:
                    pie_data["سهم (%)"] = 0.0
                    st.warning("⚠️ مجموع انتشار CO₂ برای تجهیزات انتخابی صفر است — نمودار سهم قابل رسم نیست.")
                    fig_pie = None
                # خلاصه عملکرد
                total_co2 = df_env["CO2_Total"].sum()
                target_co2 = total_co2 * (1 - reduction_target / 100)
                saved_co2 = total_co2 - target_co2
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("کل انتشار CO₂", f"{total_co2:,.0f} kg")
                with col2:
                    st.metric("هدف کاهش", f"{reduction_target}%", delta=f"-{saved_co2:,.0f} kg")
                with col3:
                    st.metric("هدف نهایی", f"{target_co2:,.0f} kg")
                with col4:
                    status = "در مسیر هدف" if total_co2 <= target_co2 else "نیاز به اقدام"
                    st.metric("وضعیت", status, delta="ISO 50001")
                if total_co2 > target_co2:
                    st.warning(f"انتشار فعلی {saved_co2:,.0f} کیلوگرم بیشتر از هدف است — نیاز به برنامه اصلاحی")
                else:
                    st.success("هدف کاهش انتشار CO₂ محقق شده — عملکرد عالی")
                # جدول سهم تجهیزات
                st.markdown("### جدول سهم تجهیزات در انتشار CO₂")
                st.dataframe(pie_data.sort_values("انتشار CO₂ (kg)", ascending=False), use_container_width=True)
                # دانلود گزارش کامل PDF
                if st.button("دانلود گزارش کامل زیست‌محیطی (PDF)", key="download_env_pdf"):
                    buffer = io.BytesIO()
                    elements = []
                    header = f"""
                    <b>گزارش زیست‌محیطی و پایداری — ردپای کربن</b><br/>
                    شرکت توسعه فراگیر سناباد<br/>
                    بازه زمانی: {safe_jalali_format(start_date)} تا {safe_jalali_format(end_date)} | تاریخ گزارش: {safe_jalali_format(pd.Timestamp.today())}
                    """
                    elements.append(Paragraph(rtl(header), ParagraphStyle('Title', fontName=FONT_NAME, fontSize=16, alignment=1, spaceAfter=30)))
                    summary = f"""
                    <b>خلاصه اجرایی:</b><br/>
                    • کل انتشار CO₂: <b>{total_co2:,.0f} کیلوگرم</b><br/>
                    • هدف کاهش: <b>{reduction_target}%</b> → مقدار هدف: <b>{target_co2:,.0f} کیلوگرم</b><br/>
                    • وضعیت: <b>{'در مسیر هدف' if total_co2 <= target_co2 else 'نیاز به اقدام فوری'}</b><br/>
                    • فاکتور انتشار: {co2_factor} kg CO₂/kWh (شبکه برق ایران)
                    """
                    elements.append(Paragraph(rtl(summary), ParagraphStyle('Normal', fontName=FONT_NAME, fontSize=12, leading=18, spaceAfter=20)))
                    # جدول سهم
                    table_data = [["تجهیز", "انتشار CO₂ (kg)", "سهم (%)"]]
                    for _, row in pie_data.iterrows():
                        table_data.append([rtl(row["تجهیز"]), f"{row['انتشار CO₂ (kg)']:,.0f}", f"{row['سهم (%)']}%"])
                    table = Table(table_data, colWidths=[200, 150, 100])
                    table.setStyle(TableStyle([
                        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#2E7D32")),
                        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
                        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                        ('FONTNAME', (0,0), (-1,-1), FONT_NAME),
                        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
                    ]))
                    elements.append(table)
                    elements.append(Spacer(1, 20))
                    # نمودارها
                    try:
                        img1 = io.BytesIO()
                        fig_trend.write_image(img1, format="png", engine="kaleido", width=1100, height=550, scale=3)
                        img1.seek(0)
                        elements.append(Image(img1, width=520, height=260))
                        if fig_pie is not None:
                            img2 = io.BytesIO()
                            fig_pie.write_image(img2, format="png", engine="kaleido", width=800, height=600, scale=3)
                            img2.seek(0)
                            elements.append(Image(img2, width=400, height=300))
                    except:
                        pass
                    footer = "تهیه‌شده توسط داشبورد پایش برق کنسانتره — توسعه فراگیر سناباد"
                    elements.append(Paragraph(rtl(footer), ParagraphStyle('Normal', fontName=FONT_NAME, fontSize=10, alignment=1, spaceBefore=30)))
                    generate_pdf("گزارش زیست‌محیطی و پایداری", elements, buffer)
                    buffer.seek(0)
                    st.success("گزارش زیست‌محیطی با موفقیت تولید شد!")
                    st.download_button(
                        label="دانلود گزارش کامل زیست‌محیطی (PDF)",
                        data=buffer.getvalue(),
                        file_name=f"زیست_محیطی_{safe_jalali_format(pd.Timestamp.today())}.pdf",
                        mime="application/pdf",
                        key="env_pdf_final"
                    )
# ===============================================
# Tab 13: استانداردها
# ===============================================
with tabs[12]:
    st.subheader("🏭 مقایسه با استانداردهای صنعتی")

    uploaded_std = st.file_uploader(
        "📂 آپلود فایل استاندارد (CSV)",
        type=["csv"],
        key="standards_uploader"
    )

    if uploaded_std:
        try:
            standards_df = pd.read_csv(uploaded_std)
            st.dataframe(standards_df)

            production = st.number_input(
                "📏 تولید کل (تن):",
                value=1000.0,
                min_value=0.01,  # جلوگیری از تقسیم بر صفر در محاسبه مصرف واقعی
                key="production_input"
            )

            selected_std = st.selectbox(
                "🔌 انتخاب تجهیز:",
                selected_equipment,
                key="std_equipment_select"
            )

            if selected_std:
                unit = get_unit_for_column(filtered_df, selected_std, st.session_state.custom_units)  # 👈 تغییر: واحد تجهیز
                # 👈 تنظیم بر اساس واحد
                if unit == "MWh":
                    actual = filtered_df[selected_std].sum() * 1000 / production # MWh به kWh
                else:
                    actual = filtered_df[selected_std].sum() / production
                st.metric(f"مصرف واقعی ({unit}/تن)", f"{actual:.2f}")

                # -------------------- مقایسه با مقدار استاندارد از فایل CSV --------------------
                std_col_name_candidates = [c for c in standards_df.columns if "تجهیز" in str(c)]
                std_value_candidates = [c for c in standards_df.columns if "استاندارد" in str(c)]
                standard_value = None
                if std_col_name_candidates and std_value_candidates:
                    name_col = std_col_name_candidates[0]
                    value_col = std_value_candidates[0]
                    match_row = standards_df[standards_df[name_col].astype(str).str.strip() == str(selected_std).strip()]
                    if not match_row.empty:
                        try:
                            standard_value = float(match_row.iloc[0][value_col])
                        except (ValueError, TypeError):
                            standard_value = None

                if standard_value is not None:
                    diff = actual - standard_value
                    diff_pct = (diff / standard_value * 100) if standard_value != 0 else None
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("مقدار استاندارد", f"{standard_value:.2f} {unit}/تن")
                    with col2:
                        st.metric("اختلاف با استاندارد", f"{diff:+.2f} {unit}/تن",
                                   delta=f"{diff_pct:+.1f}%" if diff_pct is not None else None)
                    with col3:
                        status_ok = actual <= standard_value
                        st.metric("وضعیت", "✅ مطابق استاندارد" if status_ok else "⚠️ بالاتر از استاندارد")
                    if actual <= standard_value:
                        st.success(f"مصرف واقعی «{selected_std}» ({actual:.2f}) در محدودهٔ استاندارد ({standard_value:.2f}) قرار دارد.")
                    else:
                        st.warning(f"مصرف واقعی «{selected_std}» ({actual:.2f}) از استاندارد ({standard_value:.2f}) بالاتر است — نیاز به بررسی.")
                    # نمودار مقایسه‌ای
                    fig_compare = go.Figure(data=[go.Bar(
                        x=["مصرف واقعی", "استاندارد"],
                        y=[actual, standard_value],
                        marker_color=["#C74A1B" if not status_ok else "#2E7D32", "#1A1A1A"],
                        text=[f"{actual:.2f}", f"{standard_value:.2f}"],
                        textposition="outside"
                    )])
                    fig_compare.update_layout(
                        title=f"مقایسه مصرف واقعی و استاندارد — {selected_std}",
                        yaxis_title=f"{unit}/تن",
                        template="plotly_white",
                        height=450
                    )
                    st.plotly_chart(fig_compare, use_container_width=True)
                else:
                    st.info(
                        "برای مقایسه، فایل CSV باید ستونی شامل «تجهیز» (با نام دقیقاً برابر با نام تجهیز انتخاب‌شده) "
                        "و ستونی شامل «استاندارد» داشته باشد. مقدار استاندارد منطبقی برای این تجهیز پیدا نشد؛ "
                        "فقط مصرف واقعی نمایش داده می‌شود."
                    )

                if st.button("⬇️ دانلود PDF", key="download_tab13_pdf_v1"):  # 👈 فیکس: unique key
                    if IS_CLOUD:
                        st.warning("PDF export محدود در cloud. از HTML استفاده کنید.")
                    else:
                        buffer = io.BytesIO()
                        elements = []

                        data = [standards_df.columns.tolist()] + standards_df.fillna(0).values.tolist()  # 👈 فیکس: fillna(0)
                        table = Table(data)
                        table.setStyle(TableStyle([
                            ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
                            ('GRID', (0,0), (-1,-1), 0.5, colors.lightgrey),
                            ('ALIGN', (0,0), (-1,-1), 'CENTER')
                        ]))
                        elements.append(table)

                        # خلاصه مقایسه (اگر استاندارد پیدا شده باشد)
                        if standard_value is not None:
                            elements.append(Spacer(1, 15))
                            compare_text = (
                                f"مصرف واقعی {selected_std}: {actual:.2f} {unit}/تن<br/>"
                                f"استاندارد: {standard_value:.2f} {unit}/تن<br/>"
                                f"اختلاف: {diff:+.2f} {unit}/تن ({diff_pct:+.1f}%)" if diff_pct is not None else
                                f"مصرف واقعی {selected_std}: {actual:.2f} {unit}/تن<br/>استاندارد: {standard_value:.2f} {unit}/تن"
                            )
                            elements.append(Paragraph(rtl(compare_text), ParagraphStyle('Normal', fontName=FONT_NAME, fontSize=11, alignment=1)))

                        title = f"استانداردها ({unit})"
                        if not USE_PERSIAN:
                            title = translations.get(title, title)
                        generate_pdf(title, elements, buffer)

                        st.download_button(
                            "دانلود PDF",
                            buffer.getvalue(),
                            f"مقایسه_استاندارد_{selected_std}_{safe_jalali_format(pd.Timestamp.today())}.pdf",
                            "application/pdf",
                            key="download_pdf_tab13_v1"
                        )
        except Exception as e:
            st.error(f"خطا در خواندن فایل: {e}")
    else:
        st.info("📌 فایل CSV باید ستون‌های 'تجهیز' و 'استاندارد kWh/تن' داشته باشد")
# ===============================================
# Tab 14: هزینه (اصلاح شده: سه نرخ - اوج بار، میان باری، کم باری)
# ===============================================
with tabs[13]:
    st.subheader("💰 تحلیل هزینه و بودجه")

    # فقط تجهیزات با واحد MWh/kWh در محاسبه هزینه برق لحاظ می‌شوند.
    # واحدهایی مثل تن، تومان، % یا m³ نباید به‌عنوان مصرف برق جمع زده شوند.
    energy_equipment_cost = [
        eq for eq in selected_equipment
        if get_unit_for_column(filtered_df, eq, st.session_state.custom_units) in ("MWh", "kWh")
    ]
    non_energy_equipment_cost = [eq for eq in selected_equipment if eq not in energy_equipment_cost]
    if non_energy_equipment_cost:
        st.caption(
            "⚠️ تجهیزات با واحد غیر MWh/kWh از محاسبه هزینه برق کنار گذاشته شدند (چون مصرف برق نیستند): "
            + "، ".join(non_energy_equipment_cost)
        )
    if not energy_equipment_cost:
        st.warning("هیچ‌کدام از تجهیزات انتخاب‌شده در سایدبار واحد MWh/kWh ندارند — محاسبه هزینه برق ممکن نیست.")
    else:
        # 👈 تغییر: واحد بر اساس تجهیزات انتخاب‌شده (فرض MWh برای محاسبه)
        total_consumption = 0
        for eq in energy_equipment_cost:
            eq_unit = get_unit_for_column(filtered_df, eq, st.session_state.custom_units)
            eq_cons = filtered_df[eq].sum()
            if eq_unit == "MWh":
                total_consumption += eq_cons * 1000 # به kWh
            else:
                total_consumption += eq_cons # kWh

        # سه نرخ
        rate_peak = st.number_input(
            "💸 نرخ اوج بار (تومان/kWh):",
            value=1000.0,
            key="rate_peak_input"
        )
        rate_medium = st.number_input(
            "💸 نرخ میان باری (تومان/kWh):",
            value=700.0,
            key="rate_medium_input"
        )
        rate_offpeak = st.number_input(
            "💸 نرخ کم باری (تومان/kWh):",
            value=500.0,
            key="rate_offpeak_input"
        )

        # ساعات هر دوره (مجموع 24 ساعت)
        col1, col2, col3 = st.columns(3)
        with col1:
            peak_hours = st.slider(
                "⏰ ساعات اوج بار:",
                0, 24, 8,
                key="peak_hours_slider"
            )
        with col2:
            # مقدار پیش‌فرض باید هرگز از حداکثر مجاز (24 - peak_hours) بیشتر نشود،
            # وگرنه با peak_hours بزرگ (مثلاً بیشتر از ۱۶) اسلایدر کرش می‌کند.
            max_medium = 24 - peak_hours
            default_medium = min(8, max_medium)
            medium_hours = st.slider(
                "⏰ ساعات میان باری:",
                0, max_medium, default_medium,
                key="medium_hours_slider"
            )
        with col3:
            offpeak_hours = 24 - peak_hours - medium_hours
            st.write(f"⏰ ساعات کم باری: {offpeak_hours} ساعت")

        peak_consumption = total_consumption * (peak_hours / 24)
        medium_consumption = total_consumption * (medium_hours / 24)
        offpeak_consumption = total_consumption * (offpeak_hours / 24)

        peak_cost = peak_consumption * rate_peak
        medium_cost = medium_consumption * rate_medium
        offpeak_cost = offpeak_consumption * rate_offpeak
        total_cost = peak_cost + medium_cost + offpeak_cost

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("💰 هزینه اوج بار", f"{peak_cost:,.0f} تومان")
        with col2:
            st.metric("💰 هزینه میان باری", f"{medium_cost:,.0f} تومان")
        with col3:
            st.metric("💰 هزینه کم باری", f"{offpeak_cost:,.0f} تومان")
        with col4:
            st.metric("💰 کل هزینه", f"{total_cost:,.0f} تومان")
        if st.button("⬇️ دانلود PDF", key="download_tab14_pdf_v1"):  # 👈 فیکس: unique key
            if IS_CLOUD:
                st.warning("PDF export محدود در cloud. از HTML استفاده کنید.")
            else:
                buffer = io.BytesIO()
                elements = []

                data = [["نوع", "مصرف", "هزینه (تومان)"],
                        ["اوج بار", f"{peak_consumption:.2f}", f"{peak_cost:,.0f}"],
                        ["میان باری", f"{medium_consumption:.2f}", f"{medium_cost:,.0f}"],
                        ["کم باری", f"{offpeak_consumption:.2f}", f"{offpeak_cost:,.0f}"],
                        ["کل", f"{total_consumption:.2f}", f"{total_cost:,.0f}"]]

                if not USE_PERSIAN:
                    data[0] = ["Type", "Consumption", "Cost (Toman)"]
                    data[1][0] = "Peak Load"
                    data[2][0] = "Medium Load"
                    data[3][0] = "Low Load"
                    data[4][0] = "Total"

                table = Table(data)
                table.setStyle(TableStyle([
                    ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
                    ('GRID', (0,0), (-1,-1), 0.5, colors.lightgrey),
                    ('ALIGN', (0,0), (-1,-1), 'CENTER')
                ]))
                elements.append(table)

                title = "هزینه"
                if not USE_PERSIAN:
                    title = translations.get(title, title)
                generate_pdf(title, elements, buffer)

                st.download_button(
                    "دانلود PDF",
                    buffer.getvalue(),
                    f"هزینه_برق_{safe_jalali_format(pd.Timestamp.today())}.pdf",
                    "application/pdf",
                    key="download_pdf_tab14_v1"
                )
#
# ===============================================
# Tab 15: داشبورد زنده
# ===============================================
with tabs[14]:
    st.subheader("📱 داشبورد تعاملی زنده")

    view_selector = st.selectbox(
        "🔄 انتخاب ویو:",
        ["KPI خلاصه", "روند سریع"],
        key="live_view_selector"
    )

    live_cols = st.multiselect(
        "🔌 تجهیزات:",
        selected_equipment,
        default=selected_equipment[:3] if len(selected_equipment) >= 3 else selected_equipment,
        key="live_equipment_select"
    )

    if not live_cols:
        st.info("لطفاً حداقل یک تجهیز انتخاب کنید.")
    elif view_selector == "KPI خلاصه":
            cols = st.columns(len(live_cols))
            for i, col in enumerate(live_cols):
                unit = get_unit_for_column(filtered_df, col, st.session_state.custom_units)  # 👈 تغییر: واحد هر col
                with cols[i]:
                    total = filtered_df[col].sum()
                    st.metric(f"{col}", f"{total:,.0f} {unit}")

            if st.button("⬇️ دانلود PDF", key="download_tab15_pdf_v1"):  # 👈 فیکس: unique key
                if IS_CLOUD:
                    st.warning("PDF export محدود در cloud. از HTML استفاده کنید.")
                else:
                    buffer = io.BytesIO()
                    elements = []

                    # 👈 تغییر: kpi_data با واحد
                    kpi_data = []
                    for col in live_cols:
                        unit = get_unit_for_column(filtered_df, col, st.session_state.custom_units)
                        total = filtered_df[col].sum()
                        kpi_data.append([col, f"{total:,.0f} {unit}"])
                    data = [["تجهیز", "مجموع"]] + kpi_data

                    if not USE_PERSIAN:
                        data[0] = ["Equipment", "Total"]

                    # Reshape متن‌ها برای RTL اگر فارسی (فقط رشته‌ها)
                    if USE_PERSIAN:
                        try:
                            for row in data:
                                for i, cell in enumerate(row):
                                    if isinstance(cell, str):
                                        row[i] = get_display(arabic_reshaper.reshape(cell))
                        except ImportError:
                            st.warning("برای RTL در جدول، arabic-reshaper و python-bidi رو نصب کن.")

                    table = Table(data)
                    table.setStyle(TableStyle([
                        ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
                        ('TEXTCOLOR', (0,0), (-1,0), colors.black),
                        ('GRID', (0,0), (-1,-1), 0.5, colors.lightgrey),
                        ('ALIGN', (0,0), (-1,-1), 'CENTER')
                    ]))
                    elements.append(table)

                    title = "داشبورد زنده"
                    if not USE_PERSIAN:
                        title = translations.get(title, title)
                    generate_pdf(title, elements, buffer)

                    st.download_button(
                        "دانلود PDF",
                        buffer.getvalue(),
                        f"داشبورد_زنده_{safe_jalali_format(pd.Timestamp.today())}.pdf",
                        "application/pdf",
                        key="download_pdf_tab15_v1"
                    )

    elif view_selector == "روند سریع":
            df_quick = filtered_df.copy()
            # گروه‌بندی بر اساس ماه شمسی (نه میلادی)، تا با بقیه داشبورد که همه‌جا
            # از تاریخ شمسی استفاده می‌کند هماهنگ باشد.
            df_quick["ماه"] = df_quick["تاریخ"].apply(lambda x: safe_jalali_format(x)[:7] if pd.notnull(x) else None)
            df_quick = df_quick.dropna(subset=["ماه"])
            df_quick = df_quick.groupby("ماه")[live_cols].sum().reset_index()
            df_quick = df_quick.sort_values("ماه")

            # 👈 تغییر: dynamic units برای traceها
            fig = go.Figure()
            for col in live_cols:
                unit = get_unit_for_column(filtered_df, col, st.session_state.custom_units)
                fig.add_trace(go.Scatter(
                    x=df_quick["ماه"],
                    y=df_quick[col],
                    mode="lines+markers",
                    name=f"{col} ({unit})"
                ))

            fig.update_layout(title="روند ماهانه (تقویم شمسی)", xaxis_title="ماه شمسی")
            st.plotly_chart(fig, use_container_width=True)
            if st.button("⬇️ دانلود PDF", key="download_tab15_trend_pdf_v1"):  # 👈 فیکس: unique key
                if IS_CLOUD:
                    st.warning("PDF export محدود در cloud. از HTML استفاده کنید.")
                else:
                    buffer = io.BytesIO()
                    elements = []

                    # تولید تصویر
                    img_buf = io.BytesIO()
                    fig.write_image(img_buf, format='png', width=800, height=500, scale=2)
                    img_buf.seek(0)
                    elements.append(Image(img_buf, width=500, height=300))

                    title = "داشبورد زنده - روند سریع"
                    if not USE_PERSIAN:
                        title = translations.get(title, title)
                    generate_pdf(title, elements, buffer)

                    st.download_button(
                        "دانلود PDF",
                        buffer.getvalue(),
                        f"داشبورد_زنده_روند_{safe_jalali_format(pd.Timestamp.today())}.pdf",
                        "application/pdf",
                        key="download_pdf_tab15_trend_v1"
                    )
# ===============================================
# Tab 16: گزارش سفارشی (اصلاح شده: اضافه کردن ارسال به واتساپ و تلگرام)
# ===============================================
with tabs[15]:
    st.subheader("📱 گزارش‌های سفارشی")

    include_kpi = st.checkbox("شامل KPI", key="report_include_kpi")
    include_trend = st.checkbox("شامل روند", key="report_include_trend")

    report_cols = st.multiselect(
        "🔌 تجهیزات:",
        selected_equipment,
        key="report_equipment_select"
    )

    if not report_cols:
        st.info("لطفاً حداقل یک تجهیز انتخاب کنید تا دکمهٔ تولید گزارش نمایش داده شود.")
    elif st.button("📝 تولید گزارش", key="generate_report_btn"):
        if IS_CLOUD:
            st.warning("در cloud از CSV استفاده کنید")
            df_csv_report = filtered_df[["تاریخ شمسی"] + report_cols].fillna('')
            csv = df_csv_report.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                "⬇️ دانلود CSV",
                csv.encode('utf-8-sig'),
                "report.csv",
                "text/csv",
                key="report_csv_download"
            )
        else:
            st.success("✅ گزارش آماده است")

            if include_kpi:
                st.markdown("### KPI خلاصه")
                kpi_data = filtered_df[report_cols].sum()
                # 👈 تغییر: نمایش با واحد
                for col in report_cols:
                    unit = get_unit_for_column(filtered_df, col, st.session_state.custom_units)
                    st.metric(col, f"{kpi_data[col]:,.0f} {unit}")

            if include_trend:
                st.markdown("### روند مصرف")
                # 👈 تغییر: dynamic units
                fig = go.Figure()
                for col in report_cols:
                    unit = get_unit_for_column(filtered_df, col, st.session_state.custom_units)
                    fig.add_trace(go.Scatter(
                        x=filtered_df["تاریخ"],
                        y=filtered_df[col],
                        mode="lines",
                        name=f"{col} ({unit})"
                    ))
                fig.update_layout(title="روند مصرف")
                st.plotly_chart(fig, use_container_width=True)
            # تولید PDF
            buffer = io.BytesIO()
            elements = []

            if include_kpi:
                # 👈 تغییر: kpi_list با واحد
                kpi_list = []
                for col in report_cols:
                    unit = get_unit_for_column(filtered_df, col, st.session_state.custom_units)
                    total = filtered_df[col].sum()
                    kpi_list.append([col, f"{total:,.0f} {unit}"])
                data_kpi = [["تجهیز", "مجموع"]] + kpi_list
                if not USE_PERSIAN:
                    data_kpi[0] = ["Equipment", "Total"]
                table_kpi = Table(data_kpi)
                table_kpi.setStyle(TableStyle([
                    ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
                    ('GRID', (0,0), (-1,-1), 0.5, colors.lightgrey),
                    ('ALIGN', (0,0), (-1,-1), 'CENTER')
                ]))
                elements.append(table_kpi)

            if include_trend:
                img_buf = io.BytesIO()
                fig.write_image(img_buf, format='png', width=800, height=500, scale=2)
                img_buf.seek(0)
                elements.append(Image(img_buf, width=500, height=300))

            title = "گزارش سفارشی"
            if not USE_PERSIAN:
                title = translations.get(title, title)
            generate_pdf(title, elements, buffer)

            pdf_data = buffer.getvalue()
            st.download_button(
                "⬇️ دانلود PDF",
                pdf_data,
                f"گزارش_سفارشی_{safe_jalali_format(pd.Timestamp.today())}.pdf",
                "application/pdf",
                key="download_pdf_tab16_v1"
            )

            # ارسال به واتساپ و تلگرام (لینک‌های share برای PDF)
            st.markdown("### 📱 ارسال گزارش")
            phone_number = st.text_input("شماره تلفن (برای واتساپ):", placeholder="+989123456789")
            telegram_chat = st.text_input("لینک چت تلگرام (t.me/...):", placeholder="t.me/yourchat")
            message = f"گزارش سفارشی داشبورد پایش برق: {safe_jalali_format(pd.Timestamp.now())}"  # 👈 فیکس: safe_jalali_format
            message_encoded = urllib.parse.quote(message)

            if phone_number:
                # حذف کاراکترهای غیرعددی (فاصله، خط تیره) و علامت + احتمالی از ابتدای شماره
                phone_clean = phone_number.strip().lstrip("+").replace(" ", "").replace("-", "")
                if phone_clean:
                    whatsapp_link = f"https://wa.me/{phone_clean}?text={message_encoded}"
                    st.markdown(f"[📱 ارسال به واتساپ]({whatsapp_link})")
                else:
                    st.warning("شماره تلفن معتبر نیست.")
            if telegram_chat:
                app_url = st.secrets.get('APP_URL', 'https://yourapp.streamlit.app')
                telegram_link = f"https://t.me/share/url?url={urllib.parse.quote(app_url)}&text={message_encoded}"
                st.markdown(f"[💬 ارسال به تلگرام]({telegram_link})")

with tabs[16]:
    st.subheader("🎲 شبیه‌سازی پیشرفته مصرف کل روزانه انرژی با روش مونت‌کارلو")

    # فقط تجهیزات با واحد MWh/kWh معنادار هستند؛ چون مصرف کل با واحد MWh گزارش می‌شود
    # نباید تجهیزاتی با واحد تن/تومان/% در جمع کل لحاظ شوند.
    energy_equipment_sim = [
        col for col in selected_equipment
        if get_unit_for_column(filtered_df, col, st.session_state.custom_units) in ("MWh", "kWh")
    ]
    non_energy_equipment_sim = [col for col in selected_equipment if col not in energy_equipment_sim]

    # ==================== تنظیمات و انتخاب تجهیزات داخل تب ====================
    with st.expander("⚙️ تنظیمات پیشرفته شبیه‌سازی", expanded=True):
        if non_energy_equipment_sim:
            st.caption(
                "⚠️ تجهیزات با واحد غیر MWh/kWh از فهرست انتخاب زیر حذف شده‌اند "
                "(چون مصرف کل شبیه‌سازی‌شده با واحد MWh گزارش می‌شود): "
                + "، ".join(non_energy_equipment_sim)
            )
        col_setup1, col_setup2, col_setup3, col_setup4 = st.columns(4)
        with col_setup1:
            # انتخاب تجهیزات فقط داخل این تب
            sim_cols = st.multiselect(
                "تجهیزات برای محاسبه مصرف کل و شبیه‌سازی",
                options=energy_equipment_sim,
                default=energy_equipment_sim,  # همه تجهیزات انرژی‌محور پیش‌فرض انتخاب شوند
                key="sim_eq_pro_16"
            )
        with col_setup2:
            scenario_type = st.selectbox(
                "نوع سناریو",
                ["تغییر روند ثابت", "نوسانات فصلی", "شوک‌های احتمالی", "رشد تدریجی"],
                key="scen_type_16"
            )
        with col_setup3:
            n_sims = st.select_slider("تعداد شبیه‌سازی‌ها", options=[1000, 3000, 5000, 10000], value=5000, key="nsims_16")
            noise_level = st.slider("سطح نویز و عدم قطعیت (%)", 10, 80, 30, key="noise_16") / 100
        with col_setup4:
            confidence_level = st.slider("سطح اطمینان VaR/CVaR (%)", 90, 99, 95, step=1, key="conf_16")
            exceed_threshold = st.slider("آستانه مصرف بیش از حد (برابر میانگین پایه)", 1.1, 1.5, 1.2, step=0.05, key="thresh_16")

    if not energy_equipment_sim:
        st.warning("هیچ‌کدام از تجهیزات انتخاب‌شده در سایدبار واحد MWh/kWh ندارند — شبیه‌سازی مصرف کل ممکن نیست.")
    elif not sim_cols:
        st.warning("لطفاً حداقل یک تجهیز انتخاب کنید.")
    else:
        # ==================== محاسبه داده‌های واقعی فقط بر اساس sim_cols ====================
        daily_total_real = filtered_df[sim_cols].sum(axis=1)
        daily_total_real = daily_total_real[daily_total_real > 0].dropna()

        if len(daily_total_real) < 10:
            st.error("داده کافی برای محاسبه پایه وجود ندارد (کمتر از ۱۰ روز معتبر).")
        else:
            base_mean = daily_total_real.mean()
            base_std = daily_total_real.std()
            cv_base = base_std / base_mean if base_mean > 0 else 0
            valid_days_count = len(daily_total_real)

            # ==================== کارت‌های بالا – حالا کاملاً بر اساس sim_cols ====================
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("تعداد تجهیزات انتخاب‌شده", len(sim_cols))
            with col2:
                st.metric("میانگین مصرف واقعی", f"{base_mean:.1f} MWh/روز")
            with col3:
                st.metric("انحراف معیار واقعی", f"{base_std:.1f} MWh")
            with col4:
                st.metric("داده‌های معتبر", f"{valid_days_count} روز")

            st.divider()

            # پیام سبز همخوان با کارت‌ها
            st.success(f"میانگین مصرف کل واقعی تجهیزات انتخابی: **{base_mean:.1f} MWh/روز** | انحراف معیار: **{base_std:.1f} MWh** | CV: **{cv_base:.3f}**")

            # ==================== تنظیم سناریوها ====================
            amplitude = 0
            if scenario_type == "تغییر روند ثابت":
                with st.expander("تنظیمات تغییر روند ثابت", expanded=False):
                    change = st.slider("میزان تغییر روند (%)", -40, 80, 20, key="change1") / 100
                scenarios = [0.0, change, -abs(change)/2]
                names = ["وضعیت فعلی (Base)", f"رشد {int(change*100)}%", f"کاهش {int(abs(change)/2*100)}%"]

            elif scenario_type == "نوسانات فصلی":
                with st.expander("تنظیمات نوسانات فصلی", expanded=False):
                    amplitude = st.slider("دامنه نوسان فصلی (%)", 5, 50, 20, key="amp") / 100
                scenarios = [0.0, amplitude, -amplitude]
                names = ["میانگین سالانه", "اوج مصرف (تابستان)", "کم‌مصرف (زمستان)"]

            elif scenario_type == "شوک‌های احتمالی":
                with st.expander("تنظیمات شوک‌های احتمالی", expanded=False):
                    shock_intensity = st.slider("شدت شوک حداکثر (%)", 20, 200, 80, key="shock") / 100
                scenarios = [0.0, shock_intensity/2, shock_intensity]
                names = ["عادی", "شوک متوسط", "شوک شدید"]

            else:  # رشد تدریجی
                with st.expander("تنظیمات رشد تدریجی", expanded=False):
                    monthly_growth = st.slider("رشد ماهانه (%)", 1, 15, 5, key="growth") / 100
                scenarios = [0.0, monthly_growth*6, monthly_growth*12]
                names = ["وضعیت فعلی", "پس از ۶ ماه", "پس از ۱۲ ماه"]

            # ==================== تابع شبیه‌سازی ====================
            @st.cache_data(show_spinner=False)
            def run_monte_carlo(base_mean, base_std, scenarios, names, n_sims, noise_level, scenario_type, amplitude=0):
                np.random.seed(42)
                results = {}
                for dev, name in zip(scenarios, names):
                    mean_i = base_mean * (1 + dev)
                    std_i = base_std * (1 + noise_level)
                    if scenario_type == "نوسانات فصلی" and amplitude > 0:
                        t = np.linspace(0, 4 * np.pi, n_sims)
                        seasonal = np.sin(t) * amplitude * base_mean
                        data = np.random.normal(mean_i, std_i, n_sims) + seasonal
                    else:
                        data = np.random.normal(mean_i, std_i, n_sims)
                    data = np.clip(data, 0, None)
                    results[name] = data
                return pd.DataFrame(results)

            sim_df = run_monte_carlo(
                base_mean, base_std, scenarios, names, n_sims, noise_level, scenario_type,
                amplitude=(amplitude if scenario_type == "نوسانات فصلی" else 0)
            )

            # ==================== تب‌های نتایج ====================
            t1, t2, t3, t4 = st.tabs(["📊 توزیع و مقایسه", "⚠️ تحلیل ریسک", "🔍 مقایسه سناریوها", "💾 خروجی"])

            with t1:
                colA, colB = st.columns(2)
                with colA:
                    fig_hist = px.histogram(sim_df, histnorm='probability density', nbins=70, opacity=0.75, barmode='overlay',
                                            title="توزیع احتمالاتی مصرف کل روزانه", labels={'value': 'مصرف (MWh/روز)'})
                    fig_hist.update_layout(legend_title="سناریو")
                    st.plotly_chart(fig_hist, use_container_width=True)
                with colB:
                    fig_box = px.box(sim_df, title="مقایسه جعبه‌ای سناریوها", labels={'value': 'مصرف (MWh/روز)'})
                    fig_box.update_layout(xaxis_title="سناریو")
                    st.plotly_chart(fig_box, use_container_width=True)

            with t2:
                # بدترین سناریو = بالاترین میانگین
                worst_col = sim_df.mean().idxmax()
                worst_scenario = sim_df[worst_col]
                # VaR: چون «بدترین حالت» یعنی مصرف بالا، باید صدک بالایی (confidence_level) استفاده شود
                # نه صدک پایینی (100-confidence_level)؛ در غیر این صورت CVaR عملاً میانگین بیشتر
                # توزیع می‌شود، نه میانگین دم بدترین موارد.
                var = np.percentile(worst_scenario, confidence_level)
                cvar = worst_scenario[worst_scenario >= var].mean()

                colr1, colr2, colr3, colr4 = st.columns(4)
                colr1.metric(f"VaR ({confidence_level}% اطمینان)", f"{var:.1f} MWh")
                colr2.metric("CVaR (متوسط بدترین موارد)", f"{cvar:.1f} MWh")
                colr3.metric("نوسانات میانگین سناریوها", f"{sim_df.std().mean():.1f} MWh")
                colr4.metric("CV میانگین سناریوها", f"{(sim_df.std() / sim_df.mean()).mean():.3f}")

            with t3:
                threshold_percent = int(exceed_threshold * 100)
                analysis_records = []
                for col in sim_df.columns:
                    data = sim_df[col]
                    analysis_records.append({
                        "سناریو": col,
                        "میانگین (MWh)": data.mean().round(1),
                        "انحراف معیار": data.std().round(1),
                        "CV (ریسک نسبی)": (data.std() / data.mean()).round(3),
                        "پایین‌ترین ۵٪": np.percentile(data, 5).round(1),
                        "بالاترین ۹۵٪": np.percentile(data, 95).round(1),
                        f"احتمال مصرف > {threshold_percent}% میانگین پایه": round((data > base_mean * exceed_threshold).mean() * 100, 1)
                    })
                analysis_df = pd.DataFrame(analysis_records)
                st.dataframe(analysis_df, use_container_width=True)

                # رادار
                radar_df = analysis_df.set_index("سناریو")[["میانگین (MWh)", "انحراف معیار", "CV (ریسک نسبی)"]]
                radar_norm = (radar_df - radar_df.min()) / (radar_df.max() - radar_df.min() + 1e-6)
                fig_radar = go.Figure()
                for scen in radar_norm.index:
                    fig_radar.add_trace(go.Scatterpolar(r=radar_norm.loc[scen].values, theta=radar_norm.columns, fill='toself', name=scen))
                fig_radar.update_layout(title="مقایسه راداری سناریوها (نرمال‌شده)", polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=True)
                st.plotly_chart(fig_radar, use_container_width=True)

            with t4:
                st.success("شبیه‌سازی با موفقیت انجام شد!")
                current_time_str = datetime.datetime.now().strftime('%Y%m%d_%H%M')
                csv_filename = f"MonteCarlo_TotalConsumption_{current_time_str}.csv"
                csv_data = sim_df.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
                st.download_button(label="📥 دانلود نتایج کامل (CSV)", data=csv_data, file_name=csv_filename, mime="text/csv")
                st.info(f"شامل {n_sims:,} شبیه‌سازی برای هر سناریو • داده‌های پایه: {valid_days_count} روز معتبر")

            with st.expander("📝 توضیح برای مدیر", expanded=False):
                st.markdown("""
                این تب مصرف کل تجهیزات انتخابی را با دقت بالا شبیه‌سازی می‌کند.
                - همه اعداد (کارت‌ها، پیام سبز، نمودارها) کاملاً هماهنگ و بر اساس تجهیزات انتخاب‌شده در این تب هستند.
                - تحلیل ریسک دقیق، مقایسه سناریوها و احتمال اوج‌گیری ارائه می‌شود.
                - آماده ارائه به مدیریت ارشد!
                """)
# ===============================================
# Tab 18: بهینه‌سازی (اصلاح‌شده: برای تولید کنسانتره/گندله با محدودیت‌ها)
# ===============================================
with tabs[17]:
    st.subheader("⚙️ بهینه‌سازی تولید (LP/NLP) - با محدودیت‌های واقعی")
    if not OPTIMIZATION_AVAILABLE:
        st.warning("⚠️ کتابخانه‌های بهینه‌سازی (PuLP, SciPy) نصب نیست. pip install pulp scipy رو بزن.")
    else:
        st.markdown("""
        **راهنما عملی:**  
        - **هدف:** بیشینه تولید (تن کنسانتره/گندله) یا کمینه هزینه با محدودیت‌ها (سنگ، برق، گاز، کیفیت).  
        - **مثال مدل:** تولید = 0.8 * سنگ_ورودی + 0.1 * کیفیت_Fe (خطی).  
        - **محدودیت‌ها:** تولید ≥ هدف سالانه، سنگ ≤ موجودی، برق ≤ ظرفیت، etc.  
        **فرایند:** input بدید → مدل حل → خروجی: مقادیر بهینه (سنگ، مصرف‌ها).
        """)
        
        # 👈 جدید: انتخاب نوع بهینه‌سازی (LP ساده‌تر برای تولید)
        opt_type = st.selectbox(
            "نوع بهینه‌سازی:",
            ["خطی (LP) - پیشنهادی برای تولید", "غیرخطی (NLP) - برای روابط پیچیده"],
            key="opt_type_select"
        )
        
        # 👈 جدید: انتخاب محصول (کنسانتره/گندله) و هدف تولید
        product_type = st.selectbox("محصول:", ["کنسانتره", "گندله"], key="product_select")
        target_production = st.number_input(
            f"هدف تولید سالانه ({product_type}, تن):",
            min_value=0.0,
            value=100000.0,  # مثال: 100k تن
            step=1000.0,
            key="target_production"
        )
        
        # 👈 جدید: ضرایب مدل (از دانش دامنه یا df استخراج – ساده‌سازی)
        # مثلاً: تولید = coeff_stone * سنگ + coeff_quality * Fe
        coeff_stone = st.number_input("ضریب سنگ ورودی (تن → تن تولید):", value=0.8, step=0.1, key="coeff_stone")
        coeff_quality = st.number_input("ضریب کیفیت Fe (% → تن تولید):", value=0.1, step=0.01, key="coeff_quality")
        
        # 👈 جدید: محدودیت‌ها (inputهای واقعی)
        st.markdown("### 🔒 محدودیت‌ها (موجودی/ظرفیت سالانه)")
        max_stone = st.number_input("حداکثر سنگ ورودی (تن):", value=150000.0, step=1000.0, key="max_stone")  # کمبود سنگ
        max_electricity = st.number_input("حداکثر برق (MWh):", value=50000.0, step=1000.0, key="max_electricity")  # نبود برق
        max_gas = st.number_input("حداکثر گاز (m³):", value=1000000.0, step=10000.0, key="max_gas")  # نبود گاز
        min_fe_quality = st.number_input("حداقل کیفیت Fe (%):", value=60.0, step=1.0, key="min_fe")  # مشکل کیفی
        
        # 👈 جدید: روابط مصرف (مثال: برق = 0.3 * سنگ, گاز = 5 * سنگ)
        electricity_per_stone = st.number_input("مصرف برق به ازای هر تن سنگ (MWh/تن):", value=0.3, step=0.05, key="elec_per_stone")
        gas_per_stone = st.number_input("مصرف گاز به ازای هر تن سنگ (m³/تن):", value=5.0, step=0.5, key="gas_per_stone")
        
        if opt_type == "خطی (LP) - پیشنهادی برای تولید":
            st.markdown("### 🔹 LP: بیشینه تولید با محدودیت‌ها")
            st.info("""
            **مدل خطی:**  
            max تولید = {} * سنگ + {} * Fe  
            s.t. تولید ≥ {} تن, سنگ ≤ {}, برق = {}*سنگ ≤ {}, گاز = {}*سنگ ≤ {}, Fe ≥ {}%.  
            (از PuLP با سیمپلکس).
            """.format(coeff_stone, coeff_quality, target_production, max_stone, electricity_per_stone, max_electricity, gas_per_stone, max_gas, min_fe_quality))
            
            objective_type = st.radio("هدف:", ["بیشینه تولید", "کمینه هزینه (اگر ضرایب هزینه بدید)"], key="lp_obj_type")
            sense = LpMaximize if "بیشینه" in objective_type else LpMinimize
            
            if st.button("حل مدل LP", key="solve_lp_btn"):
                # 👈 جدید: چک‌کننده پیش از حل (خارج از try – ایمن!)
                max_possible_production = coeff_stone * max_stone + coeff_quality * 100  # max با Fe=100%
                if target_production > max_possible_production:
                    st.error(f"⚠️ هدف ({target_production:,.0f} تن) > تولید max ممکن ({max_possible_production:,.0f} تن) – هدف رو کم کن!")
                else:
                    max_stone_for_elec = max_electricity / electricity_per_stone if electricity_per_stone > 0 else float('inf')
                    max_stone_for_gas = max_gas / gas_per_stone if gas_per_stone > 0 else float('inf')
                    effective_max_stone = min(max_stone, max_stone_for_elec, max_stone_for_gas)

                    if effective_max_stone < (target_production / coeff_stone):  # تقریبی بدون Fe
                        st.warning(f"⚠️ محدودیت برق/گاز سنگ رو به {effective_max_stone:,.0f} تن محدود می‌کنه – max_برق/گاز رو افزایش بده!")

                    st.info(f"💡 تولید max ممکن: {max_possible_production:,.0f} تن | سنگ موثر: {effective_max_stone:,.0f} تن")

                    try:
                        prob = LpProblem("بهینه_تولید", sense)

                        # 👈 متغیرها
                        stone_input = LpVariable("سنگ_ورودی", lowBound=0, upBound=max_stone)
                        fe_quality = LpVariable("کیفیت_Fe", lowBound=min_fe_quality, upBound=100)  # % بین min و 100

                        # 👈 تابع هدف
                        if "بیشینه تولید" in objective_type:
                            prob += coeff_stone * stone_input + coeff_quality * fe_quality, "تولید"
                        else:  # کمینه هزینه (فرض: هزینه = 1000*سنگ + 500*Fe)
                            cost_stone = st.number_input("هزینه هر تن سنگ (تومان):", value=1000.0, key="cost_stone")
                            cost_fe = st.number_input("هزینه بهبود Fe (%):", value=500.0, key="cost_fe")
                            prob += cost_stone * stone_input + cost_fe * fe_quality, "هزینه"

                        # 👈 محدودیت‌ها
                        production = coeff_stone * stone_input + coeff_quality * fe_quality
                        prob += production >= target_production, "هدف_تولید"

                        electricity_used = electricity_per_stone * stone_input
                        prob += electricity_used <= max_electricity, "محدودیت_برق"

                        gas_used = gas_per_stone * stone_input
                        prob += gas_used <= max_gas, "محدودیت_گاز"

                        # 👈 حل
                        prob.solve()

                        if LpStatus[prob.status] == "Optimal":
                            opt_stone = value(stone_input)
                            opt_fe = value(fe_quality)
                            opt_production = value(production)
                            opt_electricity = value(electricity_used)
                            opt_gas = value(gas_used)

                            st.success(f"✅ حل شد! {'تولید بهینه:' if 'بیشینه' in objective_type else 'هزینه بهینه:'} {opt_production if 'بیشینه' in objective_type else value(prob.objective):.0f}")

                            # 👈 خروجی جدول
                            results = {
                                "سنگ_ورودی (تن)": f"{opt_stone:.0f}",
                                "کیفیت_Fe (%)": f"{opt_fe:.1f}",
                                "تولید ({product_type}, تن)": f"{opt_production:.0f}",
                                "مصرف_برق (MWh)": f"{opt_electricity:.0f}",
                                "مصرف_گاز (m³)": f"{opt_gas:.0f}"
                            }
                            st.json(results)

                            results_df = pd.DataFrame(list(results.items()), columns=["پارامتر", "مقدار بهینه"])
                            st.table(results_df)

                            # نتیجه در session_state ذخیره می‌شود تا با کلیک روی دکمه دانلود PDF
                            # (که یک rerun جداگانه ایجاد می‌کند) از بین نرود — قبلاً این دکمه
                            # داخل بلوک «if st.button(حل مدل LP)» بود و با کلیک روی آن، وضعیت
                            # دکمه اصلی ریست می‌شد و نتایج محو می‌شدند.
                            st.session_state["lp_results"] = results
                            st.session_state["lp_product_type"] = product_type
                        else:
                            st.error(f"❌ حل نشد: {LpStatus[prob.status]} – محدودیت‌ها رو چک کن (مثل هدف > ظرفیت).")
                            # 👈 جدید: دیباگ – نشون بده محدودیت‌های فعال
                            for name, constraint in prob.constraints.items():
                                st.write(f"{name}: {value(constraint)}")  # slack (چقدر شل/سفت)

                    except Exception as e:
                        st.error(f"خطا در حل: {e} – PuLP رو چک کن (pip install pulp).")

            # دکمه دانلود PDF بیرون از بلوک "حل مدل LP" قرار گرفته و به session_state متکی است
            # تا با کلیک روی آن، نتایج قبلاً محاسبه‌شده از بین نروند.
            if "lp_results" in st.session_state:
                if st.button("⬇️ دانلود PDF LP", key="download_lp_pdf_v1"):
                    buffer = io.BytesIO()
                    elements = []
                    lp_results = st.session_state["lp_results"]
                    data = [["پارامتر", "مقدار بهینه"]] + [[k, v] for k, v in lp_results.items()]
                    table = Table(data)
                    table.setStyle(TableStyle([
                        ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
                        ('GRID', (0,0), (-1,-1), 0.5, colors.lightgrey),
                        ('ALIGN', (0,0), (-1,-1), 'CENTER')
                    ]))
                    elements.append(table)
                    title = f"بهینه‌سازی {st.session_state.get('lp_product_type', '')} (LP)"
                    generate_pdf(title, elements, buffer)
                    st.download_button(
                        "دانلود PDF",
                        buffer.getvalue(),
                        f"بهینه_سازی_LP_{safe_jalali_format(pd.Timestamp.today())}.pdf",
                        "application/pdf",
                        key="download_pdf_lp_final"
                    )
        
        elif opt_type == "غیرخطی (NLP) - برای روابط پیچیده":
            st.markdown("### 🔹 NLP: با روابط غیرخطی (مثل بازده کاهشی)")
            st.info("**مثال:** تولید = سنگ * (1 - 0.001 * سنگ) + Fe^2 / 100 (غیرخطی). محدودیت‌ها مثل LP.")
            
            # 👈 مشابه LP، اما با minimize (SciPy) – ساده نگه داشتم
            if st.button("حل مدل NLP", key="solve_nlp_btn"):
                try:
                    def objective(x):  # x[0]=سنگ, x[1]=Fe
                        stone, fe = x
                        return -(coeff_stone * stone * (1 - 0.001 * stone) + (fe ** 2) / 100)  # max تولید (منفی برای min)
                    
                    bounds = [(0, max_stone), (min_fe_quality, 100)]
                    constraints = [
                        {'type': 'ineq', 'fun': lambda x: coeff_stone * x[0] * (1 - 0.001 * x[0]) + (x[1] ** 2) / 100 - target_production},  # تولید >= هدف
                        {'type': 'ineq', 'fun': lambda x: max_electricity - electricity_per_stone * x[0]},  # برق
                        {'type': 'ineq', 'fun': lambda x: max_gas - gas_per_stone * x[0]}  # گاز
                    ]
                    x0 = [max_stone / 2, min_fe_quality + 10]  # نقطه شروع
                    
                    res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
                    
                    if res.success:
                        opt_stone, opt_fe = res.x
                        opt_production = -res.fun  # منفی برگردون
                        st.success(f"✅ تولید بهینه: {opt_production:.0f} تن")
                        # خروجی مشابه LP...
                        results_nlp = {
                            "سنگ_ورودی (تن)": f"{opt_stone:.0f}",
                            "کیفیت_Fe (%)": f"{opt_fe:.1f}",
                            "تولید ({product_type}, تن)": f"{opt_production:.0f}"
                        }
                        st.json(results_nlp)
                    else:
                        st.error(f"❌ {res.message}")
                except Exception as e:
                    st.error(f"خطا: {e}")
# ===============================================
# Tab 19 (index 18): شناسایی و تحلیل وابستگی‌ها (Analysis ToolPak)
# نکته: این کل بخش قبلاً بیرون از هر with tabs[...] بود و در نتیجه:
#   - تب مربوطه در رابط کاربری همیشه خالی می‌ماند.
#   - یک st.stop() سراسری داخلش (وقتی filtered_df خالی بود) کل اپ را
#     متوقف می‌کرد و تب‌های بعدی (۲۰ تا ۲۳) اصلاً رندر نمی‌شدند.
# ===============================================
with tabs[18]:
    import streamlit as st
    import pandas as pd
    import numpy as np
    import statsmodels.api as sm
    from scipy import stats
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    import plotly.express as px
    from statsmodels.formula.api import ols
    from statsmodels.stats.anova import anova_lm

    st.markdown("### 🛠️ Analysis ToolPak (سبک Excel)")
    st.caption("تحلیل آماری مشابه Excel Analysis ToolPak – مخصوص داده‌های عملیاتی")

    # ===============================
    # تابع آماده‌سازی داده (کلیدی)
    # ===============================
    def prepare_toolpak_data(df, y_col, x_cols, min_rows=10):
        cols = [y_col] + x_cols
        df2 = df[cols].copy()

        # حذف NaN
        df2 = df2.dropna()

        # حذف صفرها (عدم تولید / عدم مصرف)
        for c in cols:
            df2 = df2[df2[c] > 0]

        n = len(df2)
        return df2, n, n >= min_rows


    # ===============================
    # بررسی وجود داده
    # ===============================
    if filtered_df.empty:
        st.info("داده فیلترشده‌ای موجود نیست.")
    else:


        # ===============================
        # شناسایی ستون‌ها
        # ===============================
        numeric_cols = filtered_df.select_dtypes(include="number").columns.tolist()
        group_cols = [
            c for c in filtered_df.columns
            if filtered_df[c].dtype in ["object", "category"]
            and filtered_df[c].nunique() <= 20
        ]

        # ===============================
        # انتخاب Y و X (سبک ToolPak)
        # ===============================
        col_y, col_x = st.columns([1, 2])

        with col_y:
            target_var = st.selectbox(
                "متغیر وابسته (Y / Input Y Range)",
                ["هیچ‌کدام"] + numeric_cols
            )

        with col_x:
            if target_var != "هیچ‌کدام":
                possible_x = [c for c in numeric_cols if c != target_var]
                predictor_vars = st.multiselect(
                    "متغیرهای مستقل (X / Input X Range)",
                    possible_x,
                    default=possible_x[:1]
                )
            else:
                predictor_vars = []
                st.info("ابتدا متغیر وابسته را انتخاب کنید.")

        # ===============================
        # فعال‌سازی تحلیل‌ها
        # ===============================
        if target_var != "هیچ‌کدام" and predictor_vars:

            st.markdown("### انتخاب نوع تحلیل")

            analysis_options = [
                "هیچ‌کدام",
                "Regression (رگرسیون خطی / چندگانه)",
                "Correlation (همبستگی)",
                "VIF - بررسی هم‌خطی",
                "Descriptive Statistics"
            ]

            if group_cols:
                analysis_options.extend([
                    "ANOVA یک‌طرفه",
                    "t-test دو گروهی"
                ])

            selected_analysis = st.radio(
                "نوع تحلیل",
                analysis_options,
                horizontal=True
            )

            if selected_analysis != "هیچ‌کدام":

                # آماده‌سازی داده
                df_analysis, n_obs, valid = prepare_toolpak_data(
                    filtered_df,
                    target_var,
                    predictor_vars,
                    min_rows=10
                )

                st.caption(f"📊 تعداد داده معتبر: {n_obs}")

                if not valid:
                    st.warning("تعداد داده معتبر کمتر از ۱۰ است. تحلیل قابل اتکا نیست.")
                    st.stop()

                st.markdown("---")

                # ==================================================
                # Regression
                # ==================================================
                if selected_analysis == "Regression (رگرسیون خطی / چندگانه)":
                    X = sm.add_constant(df_analysis[predictor_vars])
                    y = df_analysis[target_var]

                    model = sm.OLS(y, X).fit()

                    st.markdown("### 📈 خروجی رگرسیون (Excel ToolPak Style)")
                    st.text(model.summary().as_text()[:3000])

                    st.success(
                        f"""
                        ✅ R² = {model.rsquared:.3f}  
                        ✅ Adjusted R² = {model.rsquared_adj:.3f}  
                        ✅ p-value مدل = {model.f_pvalue:.4f}
                        """
                    )

                    # Scatter + خط رگرسیون (اگر تک متغیره)
                    if len(predictor_vars) == 1:
                        x = predictor_vars[0]
                        fig = px.scatter(
                            df_analysis,
                            x=x,
                            y=target_var,
                            trendline="ols",
                            title="Scatter + Regression Line"
                        )
                        st.plotly_chart(fig, use_container_width=True)

                # ==================================================
                # Correlation
                # ==================================================
                elif selected_analysis == "Correlation (همبستگی)":
                    corr = (
                        df_analysis
                        .corr(method="pearson")[target_var]
                        .drop(target_var)
                        .sort_values(ascending=False)
                    )

                    st.markdown("### 🔗 ضریب همبستگی پیرسون")
                    st.dataframe(
                        corr.to_frame("Correlation")
                        .style.format("{:.3f}")
                        .background_gradient(cmap="RdYlGn")
                    )

                    fig = px.bar(
                        x=corr.index,
                        y=corr.values,
                        labels={"x": "متغیر", "y": "ضریب همبستگی"},
                        title="Correlation with Y"
                    )
                    st.plotly_chart(fig, use_container_width=True)

                # ==================================================
                # VIF
                # ==================================================
                elif selected_analysis == "VIF - بررسی هم‌خطی":
                    X = sm.add_constant(df_analysis[predictor_vars])

                    vif_df = pd.DataFrame({
                        "متغیر": X.columns,
                        "VIF": [variance_inflation_factor(X.values, i)
                                for i in range(X.shape[1])]
                    }).sort_values("VIF", ascending=False)

                    st.markdown("### ⚠️ بررسی هم‌خطی (VIF)")
                    st.dataframe(vif_df.style.format({"VIF": "{:.2f}"}))

                # ==================================================
                # Descriptive Statistics
                # ==================================================
                elif selected_analysis == "Descriptive Statistics":
                    desc = df_analysis.describe().T
                    desc["skew"] = df_analysis.skew()
                    desc["kurtosis"] = df_analysis.kurt()

                    st.markdown("### 📊 آمار توصیفی")
                    st.dataframe(desc.style.format("{:.2f}"))

                # ==================================================
                # ANOVA
                # ==================================================
                elif selected_analysis == "ANOVA یک‌طرفه":
                    group_var = st.selectbox("ستون گروه‌بندی", group_cols)
                    if group_var:
                        formula = f'Q("{target_var}") ~ C(Q("{group_var}"))'
                        model = ols(formula, data=filtered_df).fit()
                        anova_tbl = anova_lm(model)

                        st.dataframe(anova_tbl.style.format("{:.4f}"))
                        p = anova_tbl.iloc[0]["PR(>F)"]
                        st.success("✅ معنادار" if p < 0.05 else "❌ غیرمعنادار")

                # ==================================================
                # t-test
                # ==================================================
                elif selected_analysis == "t-test دو گروهی":
                    group_var = st.selectbox("ستون گروه‌بندی (۲ سطح)", group_cols)
                    if group_var:
                        groups = filtered_df[group_var].dropna().unique()
                        if len(groups) == 2:
                            g1, g2 = groups
                            a = filtered_df[filtered_df[group_var] == g1][target_var].dropna()
                            b = filtered_df[filtered_df[group_var] == g2][target_var].dropna()

                            t, p = stats.ttest_ind(a, b, equal_var=False)
                            st.write(f"t = {t:.3f} | p-value = {p:.4f}")
                            st.success("✅ تفاوت معنادار" if p < 0.05 else "❌ تفاوت غیرمعنادار")
                        else:
                            st.warning("ستون انتخابی دقیقاً دو گروه ندارد.")

        else:
            st.info("برای شروع، متغیر وابسته و حداقل یک متغیر مستقل را انتخاب کنید.")

# ============================================================
# اجرای تابع روی دیتافریم جاری (همان داده‌ای که قبلاً فیلتر شده)
# ============================================================
# فرض کنید متغیر filtered_df قبلاً در جای دیگری تعریف شده است
with tabs[19]:
    st.subheader("📊 خط مبنا (Baseline) — منطبق با شیت 're' اکسل")
    
    # ==========================================
    # کلیدهای session_state
    # ==========================================
    TAB_KEY = "tab19_"
    PRODUCTION_PELLET_KEY = f"{TAB_KEY}production_pellet"
    PRODUCTION_CONCENTRATE_KEY = f"{TAB_KEY}production_concentrate"
    
    # ==========================================
    # انتخاب تجهیزات (همان سایدبار)
    # ==========================================
    selected_baseline_equipments = selected_equipment
    if not selected_baseline_equipments:
        st.info("⚠️ لطفاً از سایدبار حداقل یک تجهیز انتخاب کنید.")
        st.stop()
    
    # ==========================================
    # انتخاب بازه پایه (برای رگرسیون)
    # ==========================================
    st.markdown("---")
    st.markdown("### 📅 بازه پایه (برای رگرسیون)")
    st.caption("داده‌های این بازه برای محاسبه معادله رگرسیون استفاده می‌شوند.")
    
    min_date_df = filtered_df["تاریخ"].min().date()
    max_date_df = filtered_df["تاریخ"].max().date()
    
    col_base1, col_base2 = st.columns(2)
    with col_base1:
        base_start = st.date_input(
            "شروع بازه پایه",
            value=max(min_date_df, max_date_df - pd.Timedelta(days=365)),
            min_value=min_date_df,
            max_value=max_date_df,
            key=f"{TAB_KEY}base_start"
        )
    with col_base2:
        base_end = st.date_input(
            "پایان بازه پایه",
            value=max_date_df - pd.Timedelta(days=30),
            min_value=min_date_df,
            max_value=max_date_df,
            key=f"{TAB_KEY}base_end"
        )
    
    if base_end < base_start:
        st.error("⚠️ تاریخ پایان نباید از تاریخ شروع کوچک‌تر باشد.")
        st.stop()
    
    st.info(f"📊 بازه پایه: **{safe_jalali_format(base_start)}** تا **{safe_jalali_format(base_end)}**")
    
    # ==========================================
    # انتخاب بازه مقایسه (برای محاسبه Residuals)
    # ==========================================
    st.markdown("---")
    st.markdown("### 📅 بازه مقایسه (برای محاسبه Residuals)")
    st.caption("برای هر ماه در این بازه، EnB با معادله رگرسیون محاسبه شده و Residuals نمایش داده می‌شود.")
    
    col_comp1, col_comp2 = st.columns(2)
    with col_comp1:
        comp_start = st.date_input(
            "شروع بازه مقایسه",
            value=max(min_date_df, max_date_df - pd.Timedelta(days=90)),
            min_value=min_date_df,
            max_value=max_date_df,
            key=f"{TAB_KEY}comp_start"
        )
    with col_comp2:
        comp_end = st.date_input(
            "پایان بازه مقایسه",
            value=max_date_df,
            min_value=min_date_df,
            max_value=max_date_df,
            key=f"{TAB_KEY}comp_end"
        )
    
    if comp_end < comp_start:
        st.error("⚠️ تاریخ پایان نباید از تاریخ شروع کوچک‌تر باشد.")
        st.stop()
    
    st.info(f"📊 بازه مقایسه: **{safe_jalali_format(comp_start)}** تا **{safe_jalali_format(comp_end)}**")
    
    # ==========================================
    # تنظیمات پیشرفته (غیرفعال‌سازی فیلترهای سخت‌گیرانه)
    # ==========================================
    st.markdown("---")
    st.markdown("### ⚙️ تنظیمات پیشرفته")
    
    disable_strict_filters = st.checkbox(
        "❌ غیرفعال‌سازی فیلترهای سخت‌گیرانه (P-Value و R²) - حالت انطباق با اکسل",
        value=True,  # پیش‌فرض فعال برای هماهنگی با اکسل
        key=f"{TAB_KEY}disable_filters"
    )
    st.caption("📌 در اکسل معمولاً فیلتر خاصی روی P-Value و R² اعمال نمی‌شود. با فعال بودن این گزینه، هر مدلی با هر R² پذیرفته می‌شود.")
    
    # ==========================================
    # دکمه اجرا
    # ==========================================
    if st.button("🚀 محاسبه و نمایش نتایج", type="primary", use_container_width=True, key=f"{TAB_KEY}run"):
        
        if not selected_baseline_equipments:
            st.warning("⚠️ لطفاً حداقل یک تجهیز انتخاب کنید.")
        else:
            # ==========================================
            # تابع کمکی: دریافت داده برای یک تجهیز در بازه به همراه متغیرهای مستقل
            # ==========================================
            def get_equip_data_with_x(df_all, equip, start_date, end_date):
                start_dt = pd.to_datetime(start_date)
                end_dt = pd.to_datetime(end_date) + pd.Timedelta(days=1)
                mask = (df_all["تاریخ"] >= start_dt) & (df_all["تاریخ"] < end_dt)
                # انتخاب ستون‌های مورد نیاز: تاریخ، مصرف، و همه ستون‌های عددی دیگر (به جز تاریخ و مصرف)
                numeric_cols = df_all.select_dtypes(include=['number']).columns.tolist()
                # حذف ستون مصرف از لیست متغیرهای مستقل (اما خود مصرف را نگه می‌داریم)
                independent_cols = [c for c in numeric_cols if c != equip]
                cols_to_select = ["تاریخ", equip] + independent_cols
                df_eq = df_all.loc[mask, cols_to_select].copy()
                df_eq = df_eq.rename(columns={equip: "consumption"})
                df_eq = df_eq.dropna(subset=["consumption"])
                df_eq = df_eq.sort_values("تاریخ").reset_index(drop=True)
                df_eq["ماه_شمسی"] = df_eq["تاریخ"].apply(lambda x: safe_jalali_format(x)[:7])
                return df_eq
            
            # ==========================================
            # حلقه روی تجهیزات
            # ==========================================
            all_results = []
            residual_summary = []
            
            for equip in selected_baseline_equipments:
                st.markdown(f"---")
                st.markdown(f"### ⚙️ تجهیز: **{equip}**")
                unit = get_unit_for_column(filtered_df, equip, st.session_state.get('custom_units', {}))
                
                # دریافت داده‌های بازه پایه (شامل همه متغیرهای مستقل)
                df_base = get_equip_data_with_x(filtered_df, equip, base_start, base_end)
                if df_base.empty:
                    st.warning(f"⚠️ داده‌ای برای تجهیز '{equip}' در بازه پایه یافت نشد.")
                    continue
                
                # دریافت داده‌های بازه مقایسه (شامل همه متغیرهای مستقل)
                df_comp = get_equip_data_with_x(filtered_df, equip, comp_start, comp_end)
                if df_comp.empty:
                    st.warning(f"⚠️ داده‌ای برای تجهیز '{equip}' در بازه مقایسه یافت نشد.")
                    continue
                
                # ==========================================
                # ۱. تعیین متغیر مستقل برای رگرسیون (بر اساس همبستگی)
                # ==========================================
                # لیست ستون‌های عددی به جز مصرف و تاریخ
                potential_x = [col for col in df_base.columns if col not in ["تاریخ", "consumption", "ماه_شمسی"]]
                
                if not potential_x:
                    st.warning(f"⚠️ هیچ متغیر مستقل عددی برای تجهیز '{equip}' در بازه پایه یافت نشد.")
                    continue
                
                # محاسبه همبستگی هر متغیر با مصرف
                corr_dict = {}
                for col in potential_x:
                    if df_base[col].nunique() > 3:  # حداقل ۴ مقدار منحصربه‌فرد برای همبستگی
                        corr = df_base["consumption"].corr(df_base[col])
                        if not np.isnan(corr):
                            corr_dict[col] = abs(corr)
                
                if not corr_dict:
                    st.warning(f"⚠️ هیچ همبستگی معنی‌داری برای تجهیز '{equip}' یافت نشد.")
                    continue
                
                # انتخاب متغیر با بیشترین همبستگی (مطلق)
                x_var = max(corr_dict, key=corr_dict.get)
                best_corr = corr_dict[x_var]
                
                # --- جدید: نمایش جدول همبستگی کامل ---
                st.markdown(f"### 📊 جدول همبستگی مصرف با متغیرهای دیگر — {equip}")
                corr_table = pd.DataFrame({
                    "متغیر": list(corr_dict.keys()),
                    "همبستگی": [round(v, 3) for v in corr_dict.values()]
                }).sort_values("همبستگی", ascending=False)
                st.dataframe(corr_table.style.background_gradient(cmap="RdBu_r", subset=["همبستگی"]), use_container_width=True, hide_index=True)
                st.caption(f"✅ متغیر انتخاب‌شده: **{x_var}** با ضریب همبستگی **{best_corr:.3f}**")
                
                # ==========================================
                # ۲. اجرای رگرسیون روی بازه پایه
                # ==========================================
                df_reg = df_base[[x_var, "consumption"]].dropna()
                if len(df_reg) < 5:
                    st.warning(f"⚠️ داده کافی برای رگرسیون در بازه پایه (حداقل ۵ نقطه) وجود ندارد.")
                    continue
                
                X = sm.add_constant(df_reg[[x_var]])
                y = df_reg["consumption"]
                
                try:
                    model = sm.OLS(y, X).fit()
                    
                    # اگر فیلتر غیرفعال باشد، هر مدلی قبول می‌شود
                    if not disable_strict_filters:
                        if model.f_pvalue >= 0.05 or model.rsquared < 0.67:
                            st.warning(f"⚠️ مدل معنی‌دار نشد (R²={model.rsquared:.3f}, p={model.f_pvalue:.4f})")
                            continue
                    
                    intercept = float(model.params.iloc[0])
                    coef = float(model.params[x_var])
                    
                    st.success(f"✅ معادله رگرسیون: EnB = {intercept:.3f} + {coef:.6f} × {x_var}")
                    st.caption(f"📊 R² = {model.rsquared:.3f} | تعداد نقاط = {len(df_reg)} | همبستگی = {best_corr:.3f}")
                    
                except Exception as e:
                    st.error(f"❌ خطا در رگرسیون: {e}")
                    continue
                
                # ==========================================
                # ۳. محاسبه EnB برای هر ماه در بازه مقایسه
                # ==========================================
                # اطمینان از وجود ستون x_var در df_comp
                if x_var not in df_comp.columns:
                    st.warning(f"⚠️ متغیر مستقل '{x_var}' در بازه مقایسه یافت نشد.")
                    continue
                
                df_comp = df_comp.dropna(subset=[x_var])
                if df_comp.empty:
                    st.warning(f"⚠️ داده‌ای برای متغیر مستقل '{x_var}' در بازه مقایسه یافت نشد.")
                    continue
                
                # محاسبه EnB برای هر ماه
                df_comp["EnB"] = intercept + coef * df_comp[x_var]
                df_comp["Residuals"] = df_comp["consumption"] - df_comp["EnB"]
                df_comp["Observation"] = range(1, len(df_comp) + 1)
                
                # ==========================================
                # ۴. نمایش جدول Residuals
                # ==========================================
                st.markdown(f"### 📋 جدول Residuals — {equip}")
                
                # ساخت جدول نمایشی
                display_df = df_comp[["Observation", "ماه_شمسی", "consumption", x_var, "EnB", "Residuals"]].copy()
                display_df.columns = ["Observation", "ماه شمسی", f"مصرف واقعی ({unit})", x_var, f"EnB ({unit})", "Residuals"]
                display_df[f"مصرف واقعی ({unit})"] = display_df[f"مصرف واقعی ({unit})"].round(2)
                display_df[f"EnB ({unit})"] = display_df[f"EnB ({unit})"].round(2)
                display_df["Residuals"] = display_df["Residuals"].round(2)
                
                st.dataframe(display_df, use_container_width=True, hide_index=True)
                
                # ==========================================
                # ۵. دانلود CSV Residuals
                # ==========================================
                residual_export = display_df.copy()
                csv_residual = residual_export.to_csv(index=False, encoding='utf-8-sig')
                st.download_button(
                    label=f"📥 دانلود Residuals برای {equip} (CSV)",
                    data=csv_residual,
                    file_name=f"Residuals_{equip}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    key=f"{TAB_KEY}download_residuals_{equip}"
                )
                
                # ==========================================
                # ۶. نمودار مقایسه مصرف واقعی و EnB  (جدید)
                # ==========================================
                st.markdown(f"### 📈 نمودار مقایسه مصرف واقعی و EnB — {equip}")
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df_comp["ماه_شمسی"],
                    y=df_comp["consumption"],
                    mode="lines+markers",
                    name=f"مصرف واقعی ({unit})",
                    line=dict(color="#C74A1B", width=3),
                    marker=dict(size=8)
                ))
                fig.add_trace(go.Scatter(
                    x=df_comp["ماه_شمسی"],
                    y=df_comp["EnB"],
                    mode="lines+markers",
                    name=f"EnB ({unit})",
                    line=dict(color="#1A1A1A", width=3, dash="dash"),
                    marker=dict(size=8, symbol="diamond")
                ))
                fig.update_layout(
                    title=f"مقایسه مصرف واقعی و EnB — {equip}",
                    xaxis_title="ماه شمسی",
                    yaxis_title=f"مقدار ({unit})",
                    height=400,
                    hovermode="x unified",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # ==========================================
                # ۷. خلاصه آماری Residuals
                # ==========================================
                avg_res = df_comp["Residuals"].mean()
                max_res = df_comp["Residuals"].max()
                min_res = df_comp["Residuals"].min()
                above_count = (df_comp["Residuals"] > 0).sum()
                below_count = (df_comp["Residuals"] < 0).sum()
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("میانگین Residuals", f"{avg_res:+.2f} {unit}")
                with col2:
                    st.metric("بیشترین Residuals", f"{max_res:+.2f} {unit}", delta="بالاترین")
                with col3:
                    st.metric("کمترین Residuals", f"{min_res:+.2f} {unit}", delta="پایین‌ترین")
                with col4:
                    st.metric("تعداد", f"{below_count} منفی / {above_count} مثبت")
                
                # ==========================================
                # ۸. ذخیره نتایج برای دانلود خلاصه
                # ==========================================
                residual_summary.append({
                    "تجهیز": equip,
                    "متغیر مستقل": x_var,
                    "همبستگی": round(best_corr, 3),
                    "Intercept": round(intercept, 3),
                    "ضریب": round(coef, 6),
                    "R²": round(model.rsquared, 3),
                    "تعداد نقاط": len(df_reg),
                    "میانگین Residuals": round(avg_res, 2),
                    "بیشترین Residuals": round(max_res, 2),
                    "کمترین Residuals": round(min_res, 2)
                })
            
            # ==========================================
            # ۹. دانلود خلاصه نتایج
            # ==========================================
            if residual_summary:
                st.markdown("---")
                st.markdown("## 📥 خروجی خلاصه")
                summary_df = pd.DataFrame(residual_summary)
                st.dataframe(summary_df, use_container_width=True, hide_index=True)
                
                csv_summary = summary_df.to_csv(index=False, encoding='utf-8-sig')
                st.download_button(
                    label="📥 دانلود خلاصه نتایج (CSV)",
                    data=csv_summary,
                    file_name=f"Baseline_Summary_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    key=f"{TAB_KEY}download_summary"
                )
                
                st.success(f"✅ محاسبه برای {len(residual_summary)} تجهیز با موفقیت انجام شد.")
                st.balloons()
    
    # ==========================================
    # راهنمای استفاده
    # ==========================================
    with st.expander("ℹ️ راهنمای استفاده — منطبق با شیت 're' اکسل", expanded=False):
        st.markdown("""
        ### 🎯 هدف این تب
        
        این تب دقیقاً مانند **شیت "re"** در فایل اکسل شما عمل می‌کند:
        
        1. **بازه پایه** را انتخاب کنید. داده‌های این بازه برای **محاسبه معادله رگرسیون** استفاده می‌شوند.
        2. **متغیر مستقل** به‌صورت خودکار بر اساس **بیشترین همبستگی** با مصرف در بازه پایه انتخاب می‌شود.
        3. **بازه مقایسه** را انتخاب کنید. برای هر ماه در این بازه، **EnB با همان معادله رگرسیون** محاسبه می‌شود.
        4. **جدول Residuals** شامل Observation, مصرف واقعی, EnB, و Residuals نمایش داده می‌شود.
        5. **نمودار مقایسه** مصرف واقعی و EnB برای تحلیل بصری.
        6. **جدول همبستگی** کامل برای توجیه انتخاب متغیر مستقل.
        
        ### 🔧 تنظیمات
        
        - **غیرفعال‌سازی فیلترهای سخت‌گیرانه**: برای هماهنگی با اکسل، این گزینه را فعال بگذارید (پیش‌فرض فعال است).
        
        ### 📊 خروجی
        
        - جدول همبستگی کامل
        - معادله رگرسیون
        - جدول Residuals برای هر تجهیز
        - نمودار مقایسه
        - خلاصه آماری Residuals
        - امکان دانلود CSV برای هر تجهیز و خلاصه کلی
        
        ### 📌 نکته مهم
        
        برای بازتولید دقیق خروجی اکسل، بازه پایه و بازه مقایسه را **همانند اکسل** تنظیم کنید (معمولاً بازه پایه همان داده‌های رگرسیون و بازه مقایسه همان داده‌هایی است که Residuals برای آن‌ها محاسبه شده است).
        """)
with tabs[20]:
    st.subheader("📋 گزارش EnMS برای ISO 50001 (چرخه PDCA بهبودیافته)")
    st.markdown("""
    **راهنما:** چرخه PDCA برای مدیریت انرژی مداوم (ISO 50001). هر فاز رو چک کن و شواهد رو ببین.
    - **پیشرفت کلی:** محاسبه خودکار بر اساس داده‌های داشبورد (e.g., KPIها، Baseline).
    """)

    # ===============================================
    # 📑 خط مبناهای رسمی/ممیزی‌شده (ثبت‌شده از تب «خط مبنا»)
    # ===============================================
    st.markdown("### 📑 خط مبناهای رسمی ثبت‌شده (Approved EnB)")
    OFFICIAL_BASELINE_FILE_PDCA = "official_baselines.csv"
    if os.path.exists(OFFICIAL_BASELINE_FILE_PDCA):
        try:
            official_df = pd.read_csv(OFFICIAL_BASELINE_FILE_PDCA)
        except Exception:
            official_df = pd.DataFrame()
    else:
        official_df = pd.DataFrame()

    if official_df.empty:
        st.info(
            "⚠️ هنوز هیچ خط مبنایی به‌عنوان «رسمی» ثبت نشده است. "
            "برای ثبت، به تب «📊 خط مبنا (Baseline)» برو، حالت «رگرسیون رسمی» رو انتخاب کن "
            "و پس از محاسبه، دکمه‌ی «🔒 ثبت به‌عنوان خط مبنای رسمی» رو بزن."
        )
    else:
        active_df = official_df[official_df["وضعیت"] == "فعال"] if "وضعیت" in official_df.columns else official_df
        if not active_df.empty:
            st.success(f"✅ {len(active_df)} خط مبنای رسمی فعال ثبت شده است.")
            st.dataframe(active_df, use_container_width=True, hide_index=True)
        else:
            st.warning("خط مبنای «فعال» یافت نشد (ممکن است همه در آرشیو باشند).")

        with st.expander("📜 تاریخچه‌ی کامل خط مبناها (شامل نسخه‌های آرشیو‌شده)", expanded=False):
            st.dataframe(official_df, use_container_width=True, hide_index=True)
            csv_official = official_df.to_csv(index=False, encoding="utf-8-sig")
            st.download_button(
                "📥 دانلود کامل تاریخچه خط مبناها",
                data=csv_official,
                file_name="official_baselines_history.csv",
                mime="text/csv",
                key="pdca_download_official_baselines"
            )

    # محاسبه پیشرفت خودکار (ادغام واقعی با داده‌های داشبورد)
    # مصرف کل بر اساس تجهیزات انرژی‌محور انتخاب‌شده در سایدبار (نه یک نام ستون ثابت
    # که در داده‌های واقعی احتمالاً اصلاً وجود ندارد)
    energy_equipment_pdca = [
        col for col in selected_equipment
        if get_unit_for_column(filtered_df, col, st.session_state.custom_units) in ("MWh", "kWh")
    ]
    total_kpi = 0.0
    for col in energy_equipment_pdca:
        unit = get_unit_for_column(filtered_df, col, st.session_state.custom_units)
        val = filtered_df[col].sum()
        total_kpi += val if unit == "MWh" else val / 1000  # همه به MWh

    # آیا Baseline واقعاً در تب «خط مبنا» محاسبه و ذخیره شده؟ (کلید واقعی همان تب)
    baseline_set = bool(st.session_state.get("tab19_baseline_results"))

    goal_reduction = st.number_input("هدف کاهش مصرف (%):", value=15.0, key="pdca_goal")

    # پیشرفت واقعی: فقط وقتی قابل محاسبه است که Baseline واقعی موجود باشد
    # (مقایسه میانگین مصرف فعلی نسبت به میانگین مصرف Baseline). در غیر این صورت
    # به‌جای یک عدد جعلی/ثابت، صراحتاً «نامشخص» نمایش داده می‌شود.
    current_improvement = None
    if baseline_set and energy_equipment_pdca:
        baseline_results = st.session_state.get("tab19_baseline_results", {})
        saving_percents = [
            baseline_results[col]["saving_percent"]
            for col in energy_equipment_pdca
            if col in baseline_results and "saving_percent" in baseline_results[col]
        ]
        if saving_percents:
            current_improvement = sum(saving_percents) / len(saving_percents)

    improvement_text = f"{current_improvement:.1f}%" if current_improvement is not None else "نامشخص (ابتدا در تب «خط مبنا» Baseline را محاسبه کنید)"

    # PDCA با expander و شواهد
    pdca_phases = {
        "Plan (برنامه‌ریزی)": {
            "عناصر": ["سیاست انرژی", "اهداف و KPIها", "اقدام‌برنامه‌ها"],
            "status": st.checkbox("کامل", key="plan_cb"),
            "evidence": f"اهداف: {goal_reduction}% کاهش | KPI کل: {total_kpi:,.0f} MWh"
        },
        "Do (اجرا)": {
            "عناصر": ["آموزش کارکنان", "پیاده‌سازی اقدام‌ها", "پایش عملیاتی"],
            "status": st.checkbox("در حال اجرا", key="do_cb"),
            "evidence": "اقدام‌ها: داشبورد KPI + آموزش (لینک به تب ۶)"
        },
        "Check (نظارت)": {
            "عناصر": ["اندازه‌گیری عملکرد", "تحلیل ناهنجاری", "audit داخلی"],
            "status": st.checkbox("نظارت فعال", key="check_cb"),
            "evidence": f"Baseline تنظیم: {'✅ بله' if baseline_set else '❌ خیر — به تب «خط مبنا» مراجعه کنید'} | برای بررسی ناهنجاری‌ها به تب «ناهنجاری‌ها» مراجعه کنید"
        },
        "Act (اقدام)": {
            "عناصر": ["بررسی مدیریت", "بهبود مداوم", "تصحیح انحرافات"],
            "status": st.checkbox("بهبود اعمال", key="act_cb"),
            "evidence": f"پیشرفت واقعی: {improvement_text} | هدف: {goal_reduction}% | بهینه‌سازی (تب ۱۸)"
        }
    }

    # نمایش expanderها
    pdca_summary = []
    for phase, details in pdca_phases.items():
        with st.expander(f"🔄 {phase} ({'✅' if details['status'] else '❌'} )"):
            st.markdown("**عناصر کلیدی (ISO 50001):**")
            for elem in details["عناصر"]:
                st.checkbox(elem, key=f"{phase}_{elem}")
            st.info(details["evidence"])

        pdca_summary.append({"فاز": phase, "Status": details["status"], "شواهد": details["evidence"]})

    # جدول خلاصه PDCA (خروجی اصلی)
    pdca_df = pd.DataFrame(pdca_summary)
    st.subheader("📊 جدول خلاصه PDCA")
    st.dataframe(pdca_df.style.applymap(lambda x: 'background-color: lightgreen' if x else 'background-color: lightcoral', subset=['Status']))

    # چارت پیشرفت (radar برای PDCA)
    import plotly.graph_objects as go
    categories = [phase for phase, _ in pdca_phases.items()]
    values = [1 if details['status'] else 0 for _, details in pdca_phases.items()]
    fig_radar = go.Figure(data=go.Scatterpolar(r=[v * 100 for v in values], theta=categories, fill='toself'))
    fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), title="چارت پیشرفت PDCA (%)")
    st.plotly_chart(fig_radar, use_container_width=True)

    # Metric کلی
    pdca_completion = sum([1 for d in pdca_summary if d['Status']]) / len(pdca_summary) * 100
    st.metric("پیشرفت کلی PDCA", f"{pdca_completion:.0f}%", delta=pdca_completion - 100)

    # Exportها (خروجی غنی)
    col1, col2 = st.columns(2)
    with col1:
        csv_pdca = pdca_df.to_csv(index=False, encoding='utf-8-sig')
        st.download_button("⬇️ دانلود CSV PDCA", csv_pdca, "pdca_summary.csv", "text/csv")
    with col2:
        if st.button("⬇️ دانلود PDF EnMS کامل"):
            buffer = io.BytesIO()
            elements = []
            # جدول PDCA
            data = [["فاز", "Status", "شواهد"]] + pdca_df.values.tolist()
            table = Table(data)
            table.setStyle(TableStyle([('BACKGROUND', (0,0), (-1,0), colors.lightgrey), ('GRID', (0,0), (-1,-1), 1, colors.black)]))
            elements.append(table)
            # چارت به عنوان تصویر
            img_buf = io.BytesIO()
            fig_radar.write_image(img_buf, format='png', width=600, height=400)
            img_buf.seek(0)
            elements.append(Image(img_buf, width=400, height=300))
            # خلاصه
            elements.append(Paragraph(f"پیشرفت کلی: {pdca_completion:.0f}% | هدف: {goal_reduction}%", getSampleStyleSheet()['Normal']))
            generate_pdf("گزارش PDCA EnMS ISO 50001", elements, buffer)
            st.download_button("دانلود PDF", buffer.getvalue(), "enms_pdca_report.pdf", "application/pdf")

    st.caption("💡 بر اساس ISO 50001: PDCA iterative برای بهبود مداوم. برای audit، شواهد رو مستند کن.")
# ==================== Tab جدید: فرمول‌ها و ممیزی ====================
with tabs[21]:  # یا index جدید
    st.subheader("📋 فرمول‌های محاسباتی برای ممیزی انرژی")
    st.markdown("""
    **هدف این تب:** مستندسازی تمام فرمول‌های استفاده‌شده در داشبورد برای ردیابی و ممیزی (ISO 50001).
    - هر فرمول شامل: توضیح، فرمول ریاضی، ورودی/خروجی، و منبع.
    - **نسخه داشبورد:** v1.2 | تاریخ بروزرسانی: {now} | مسئول: [نام شما]
    """.format(now=pd.Timestamp.now().strftime('%Y/%m/%d')))

    # دیکشنری فرمول‌ها (از کد فعلی استخراج‌شده – گسترش بده)
    formulas_data = {
        "KPI میانگین مصرف": {
            "فرمول": "میانگین = Σ(مصرف_i) / n",
            "توضیح": "میانگین مصرف تجهیز در بازه زمانی (روزانه/ماهانه).",
            "ورودی‌ها": "ستون تجهیز (e.g., مصرف برق [MWh]), تعداد رکوردها (n)",
            "خروجی": "مقدار میانگین (MWh)",
            "منبع": "داده‌های اکسل + pandas.mean() | ISO 50001 - KPI Tracking"
        },
        "EnPI (شاخص عملکرد انرژی)": {
            "فرمول": "EnPI = (Σ(kWh) / Σ(تن تولید))",
            "توضیح": "مصرف انرژی به ازای واحد تولید (kWh/تن).",
            "ورودی‌ها": "مصرف کل (kWh = MWh * 1000), تولید کل (تن)",
            "خروجی": "EnPI (kWh/تن)",
            "منبع": "ISO 50001 - EnPI Calculation | کد: filtered_df[col].sum() / total_production"
        },
        "Baseline اتوماتیک (Specific)": {
            "فرمول": "Baseline = median(consumption / regressor) * mean(regressor)",
            "توضیح": "خط مبنا بر اساس مصرف ویژه (per unit production) + میانگین تولید.",
            "ورودی‌ها": "consumption (مصرف), regressor (e.g., تولید [تن]), بازه پایه (365 روز)",
            "خروجی": "Baseline (MWh)",
            "منبع": "روش پیشنهادی ISO 50001 برای Baseline | کد: compute_auto_baseline() با IsolationForest برای outliers"
        },
        "ناهنجاری (IQR)": {
            "فرمول": "UCL = Q3 + 1.5*IQR | LCL = max(0, Q1 - 1.5*IQR)",
            "توضیح": "تشخیص outliers با Tukey method (non-parametric).",
            "ورودی‌ها": "داده‌های غیرصفر مصرف",
            "خروجی": "نقاط outlier (بالای UCL)",
            "منبع": "ASTM E2587 - Energy Audit | کد: non_zero_data.quantile()"
        },
        "پیش‌بینی ML (Prophet)": {
            "فرمول": "y(t) = g(t) + s(t) + h(t) + ε_t  (trend + seasonal + holiday)",
            "توضیح": "مدل additive برای time-series forecasting.",
            "ورودی‌ها": "ds (تاریخ), y (مصرف), train/test split (80/20)",
            "خروجی": "yhat (پیش‌بینی) + CI (اعتماد 95%)",
            "منبع": "Facebook Prophet Docs | کد: m.fit(train_df); m.predict(future)"
        },
        "بهینه‌سازی LP (PuLP)": {
            "فرمول": "max Z = c1*x1 + c2*x2 | s.t. A*x ≤ b, x ≥ 0",
            "توضیح": "Linear Programming برای بیشینه تولید با محدودیت (سنگ, برق).",
            "ورودی‌ها": "ضرایب (e.g., 0.8*سنگ), محدودیت‌ها (max_stone=150k تن)",
            "خروجی": "مقادیر بهینه (x1=سنگ, Z=تولید)",
            "منبع": "ISO 50001 - Optimization | کد: LpProblem() + .solve()"
        }
        # اضافه کن: فرمول‌های بیشتر مثل CO2 (kg = MWh * 1000 * 0.5), Waterfall تغییرات, etc.
    }

    # تبدیل به DataFrame برای نمایش
    formulas_df = pd.DataFrame.from_dict(formulas_data, orient='index')
    st.dataframe(formulas_df, use_container_width=True, height=400)

    # فیلتر تعاملی (اختیاری)
    selected_formula = st.selectbox("فرمول خاص:", options=list(formulas_data.keys()))
    if selected_formula:
        st.markdown(f"**جزئیات {selected_formula}:**")
        # نکته: از st.latex استفاده نمی‌شود چون رشتهٔ فرمول شامل کلمات فارسی
        # (مثل «میانگین»، «مصرف_i») است که بدون \text{...} در LaTeX به‌شکل
        # بهم‌ریخته و غیرقابل‌خواندن رندر می‌شود. st.code خوانایی بهتری می‌دهد.
        st.code(formulas_data[selected_formula]["فرمول"], language=None)
        st.write(f"**توضیح:** {formulas_data[selected_formula]['توضیح']}")
        st.write(f"**ورودی/خروجی:** {formulas_data[selected_formula]['ورودی‌ها']} → {formulas_data[selected_formula]['خروجی']}")
        st.caption(f"**منبع:** {formulas_data[selected_formula]['منبع']}")

    # Exportها
    col1, col2 = st.columns(2)
    with col1:
        # CSV ساده
        csv = formulas_df.to_csv(index_label="فرمول", encoding='utf-8-sig')
        st.download_button("⬇️ دانلود CSV (برای Excel)", csv, "energy_audit_formulas.csv", "text/csv")
    with col2:
        # PDF (با ReportLab)
        if st.button("⬇️ دانلود PDF ممیزی"):
            buffer = io.BytesIO()
            elements = []
            header = f"""
            <b>فرمول‌های محاسباتی برای ممیزی انرژی</b><br/>
            نسخه داشبورد: v1.2 | تاریخ گزارش: {safe_jalali_format(pd.Timestamp.today())}
            """
            elements.append(Paragraph(rtl(header), ParagraphStyle('Title', fontName=FONT_NAME, fontSize=13, alignment=1, spaceAfter=20)))
            # جدول واقعی فرمول‌ها (قبلاً ساخته نمی‌شد و PDF کاملاً خالی بود)
            table_headers = ["فرمول"] + list(formulas_df.columns)
            table_data = [table_headers]
            for formula_name, row in formulas_df.iterrows():
                table_data.append([rtl(str(formula_name))] + [rtl(str(row[c])) for c in formulas_df.columns])
            audit_table = Table(table_data, repeatRows=1)
            audit_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#C74A1B")),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, -1), FONT_NAME),
                ('FONTSIZE', (0, 0), (-1, -1), 7),
                ('GRID', (0, 0), (-1, -1), 0.4, colors.grey),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ]))
            elements.append(audit_table)
            generate_pdf("فرمول‌های ممیزی انرژی", elements, buffer)
            st.download_button(
                "دانلود PDF",
                buffer.getvalue(),
                f"فرمول_های_ممیزی_{safe_jalali_format(pd.Timestamp.today())}.pdf",
                "application/pdf",
                key="download_pdf_audit_formulas"
            )

    # نکته اضافی: لینک به استاندارد
    st.markdown("""
    **نکته ممیزی:** این فرمول‌ها با ISO 50001 همخوانی دارن. برای audit کامل، baseline رو هر ۱ سال بروز کن (بازه ۳۶۵ روز).
    """)

# ===============================================
# Tab 22: برنامه عملیاتی - نسخه نهایی و حرفه‌ای
# جدول کامل + فیلتر داخلی + ویرایش و حذف در هر ردیف
# ===============================================
# فقط این کد رو داخل with tabs[22]: بذار
# ===============================================
with tabs[22]:
    st.markdown("""
    <div style="background: linear-gradient(90deg, #C74A1B, #561018); padding: 20px; border-radius: 15px; text-align: center; color: white; font-size: 28px; font-weight: bold; margin-bottom: 30px;">
        برنامه عملیاتی کاهش مصرف انرژی
    </div>
    """, unsafe_allow_html=True)

    # --- فایل ذخیره ---
    ACTION_PLAN_FILE = "action_plan.csv"
    if 'action_plan' not in st.session_state:
        if os.path.exists(ACTION_PLAN_FILE):
            st.session_state.action_plan = pd.read_csv(ACTION_PLAN_FILE)
        else:
            st.session_state.action_plan = pd.DataFrame(columns=[
                'شناسه', 'تاریخ_ثبت', 'اقدام', 'تجهیز_منطقه', 'صرفهجویی_تخمینی_MWh_ماه',
                'هزینه_اجرا_میلیون_تومان', 'مسئول', 'وضعیت'
            ])

    # نکته: اینجا عمداً از نام «action_df» استفاده می‌شود (نه «filtered_df»)
    # چون «filtered_df» در سراسر داشبورد برای داده مصرف تجهیزات استفاده می‌شود
    # و همنام کردن این دو، برای هر کد آینده‌ای که بعد از این تب اضافه شود تله ایجاد می‌کند.
    action_df = st.session_state.action_plan.copy()

    # --- KPI ---
    total_est = action_df['صرفهجویی_تخمینی_MWh_ماه'].sum()
    col1, col2, col3 = st.columns(3)
    with col1: st.metric("تعداد اقدامات", len(action_df))
    with col2: st.metric("صرفه‌جویی تخمینی", f"{total_est:,.0f} MWh/ماه")
    with col3: st.metric("وضعیت کلی", "فعال" if len(action_df)>0 else "خالی")

    st.markdown("---")

    status_colors = {
        "همه": "#6c757d", "در حال بررسی": "#ffc107", "تصویب شده": "#007bff",
        "در حال اجرا": "#fd7e14", "انجام شد": "#28a745", "متوقف شد": "#dc3545"
    }
    status_bg_colors = {
        "در حال بررسی": "#fff3cd", "تصویب شده": "#d1ecf1", "در حال اجرا": "#ffe5d0",
        "انجام شد": "#d4edda", "متوقف شد": "#f8d7da"
    }

    # --- فیلتر با دکمه‌های رنگی ---
    st.subheader("فیلتر بر اساس وضعیت")
    statuses = ["همه"] + sorted(action_df['وضعیت'].unique().tolist()) if not action_df.empty else ["همه"]
    selected = st.session_state.get("filter_status", "همه")

    btn_cols = st.columns(len(statuses))
    for i, status in enumerate(statuses):
        with btn_cols[i]:
            if st.button(
                status,
                key=f"filter_{status}",
                use_container_width=True,
                type="primary" if selected == status else "secondary"
            ):
                st.session_state.filter_status = status
                st.rerun()

    # نکته: عمداً reset_index انجام نمی‌شود تا ایندکس هر ردیف فیلترشده،
    # دقیقاً همان ایندکس اصلی‌اش در action_df/session_state.action_plan بماند.
    # قبلاً با reset_index(drop=True)، ایندکس صفرمبنای زیرمجموعه فیلترشده به‌اشتباه
    # مستقیماً برای ویرایش/حذف در دیتافریم کامل استفاده می‌شد و هر وقت فیلتر فعال
    # بود، ممکن بود ردیف کاملاً اشتباهی ویرایش یا حذف شود.
    filtered_action_df = action_df if selected == "همه" else action_df[action_df['وضعیت'] == selected].copy()

    # --- جدول حرفه‌ای ---
    st.subheader(f"اقدامات ({selected})")

    if not filtered_action_df.empty:
        for idx in filtered_action_df.index:
            row = filtered_action_df.loc[idx]

            # رنگ‌های متناظر با وضعیت همین ردیف (نه رنگ نشت‌کرده از حلقه فیلتر بالا)
            row_bg_color = status_bg_colors.get(row['وضعیت'], "#f8f9fa")
            row_border_color = status_colors.get(row['وضعیت'], "#6c757d")

            col_info, col_edit, col_delete = st.columns([10, 1, 1])
            with col_info:
                st.markdown(f"""
                <div style="background-color: {row_bg_color}; padding: 15px; border-radius: 12px; margin: 5px 0; border-left: 6px solid {row_border_color}; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
                    <h4 style="margin:0; color:#1a1a1a;">{row['اقدام'][:80]}{'...' if len(str(row['اقدام']))>80 else ''}</h4>
                    <p style="margin:5px 0; color:#555;">
                        <strong>تجهیز:</strong> {row['تجهیز_منطقه']} |
                        <strong>صرفه‌جویی:</strong> {row['صرفهجویی_تخمینی_MWh_ماه']:,.0f} MWh/ماه |
                        <strong>هزینه:</strong> {row['هزینه_اجرا_میلیون_تومان']:,.0f} میلیون تومان |
                        <strong>مسئول:</strong> {row['مسئول']}
                    </p>
                    <div style="text-align:left; margin-top:10px; font-size:14px; color:#1a1a1a;">
                        وضعیت: <strong style="color:{row_border_color};">{row['وضعیت']}</strong> |
                        ثبت: {row['تاریخ_ثبت']} | شناسه: {row['شناسه']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            with col_edit:
                if st.button("✏️", key=f"edit_real_{idx}", help="ویرایش این اقدام"):
                    st.session_state.edit_idx = idx
                    st.rerun()
            with col_delete:
                if st.button("🗑️", key=f"delete_real_{idx}", help="حذف این اقدام"):
                    st.session_state.action_plan = action_df.drop(idx).reset_index(drop=True)
                    st.session_state.action_plan.to_csv(ACTION_PLAN_FILE, index=False, encoding='utf-8-sig')
                    st.success("اقدام حذف شد")
                    st.rerun()

        # فرم ویرایش
        if "edit_idx" in st.session_state:
            idx = st.session_state.edit_idx
            if idx in action_df.index:
                row = action_df.loc[idx]
                st.markdown("### ویرایش اقدام")
                with st.form("edit_form"):
                    action = st.text_area("اقدام", value=row['اقدام'])
                    area = st.text_input("تجهیز/منطقه", value=row['تجهیز_منطقه'])
                    saving = st.number_input("صرفه‌جویی", value=float(row['صرفهجویی_تخمینی_MWh_ماه']))
                    cost = st.number_input("هزینه", value=float(row['هزینه_اجرا_میلیون_تومان']))
                    status = st.selectbox("وضعیت", ["در حال بررسی", "تصویب شده", "در حال اجرا", "انجام شد", "متوقف شد"],
                                        index=["در حال بررسی", "تصویب شده", "در حال اجرا", "انجام شد", "متوقف شد"].index(row['وضعیت']))

                    if st.form_submit_button("ذخیره تغییرات", type="primary"):
                        action_df.loc[idx] = [row['شناسه'], row['تاریخ_ثبت'], action, area, saving, cost, row['مسئول'], status]
                        st.session_state.action_plan = action_df
                        action_df.to_csv(ACTION_PLAN_FILE, index=False, encoding='utf-8-sig')
                        del st.session_state.edit_idx
                        st.success("تغییرات ذخیره شد")
                        st.rerun()
            else:
                del st.session_state.edit_idx

    else:
        st.info("هیچ اقدامی ثبت نشده است.")

    st.markdown("---")

    # --- فرم ثبت جدید ---
    st.subheader("ثبت اقدام جدید")
    with st.form("new_action"):
        col1, col2 = st.columns(2)
        with col1:
            action = st.text_area("شرح اقدام", height=100)
            area = st.text_input("تجهیز / منطقه")
            saving = st.number_input("صرفه‌جویی تخمینی (MWh/ماه)", value=50.0)
        with col2:
            cost = st.number_input("هزینه اجرا (میلیون تومان)", value=10.0)
            status = st.selectbox("وضعیت", ["در حال بررسی", "تصویب شده", "در حال اجرا"])

        if st.form_submit_button("ثبت اقدام جدید", type="primary", use_container_width=True):
            new_id = f"ACT-{len(action_df)+1:04d}"
            new_row = pd.DataFrame([{
                'شناسه': new_id, 'تاریخ_ثبت': JalaliDate.today().strftime('%Y/%m/%d'),
                'اقدام': action, 'تجهیز_منطقه': area, 'صرفهجویی_تخمینی_MWh_ماه': saving,
                'هزینه_اجرا_میلیون_تومان': cost, 'مسئول': "نامشخص", 'وضعیت': status
            }])
            st.session_state.action_plan = pd.concat([action_df, new_row], ignore_index=True)
            st.session_state.action_plan.to_csv(ACTION_PLAN_FILE, index=False, encoding='utf-8-sig')
            st.success("اقدام جدید ثبت شد!")
            st.balloons()
            st.rerun()