import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    precision_recall_curve
)
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow.keras import layers, models
import random
import os

df = pd.read_csv("C:/Users/MICRON PRO/Desktop/synthetic_liver_cancer_dataset.csv")
print(df)

df.info()  
print(df.isnull().sum())
print(df.duplicated().sum())
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
print(df.describe().T)

def define_outliers(df):
    outliers = pd.DataFrame()
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outlier = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outliers = pd.concat([outliers, outlier])

    outliers.drop_duplicates(inplace=True)
    print(f'Liczba wartości odstających w zestawie danych wynosi {len(outliers)}')

outliers = define_outliers(df)

print(df.columns)
print(df.nunique())

# Wykresy wartości odstających
continuous_features = ['age', 'bmi', 'liver_function_score', 'alpha_fetoprotein_level']

tit = {
    'age': 'Rozkład wieku pacjentów',
    'bmi': 'Rozkład wskaźnika masy ciała (BMI)',
    'liver_function_score': 'Rozkład wyniku czynności wątroby',
    'alpha_fetoprotein_level': 'Rozkład poziomu AFP'
}

y_labels = {
    'age': 'Wiek',
    'bmi': 'BMI',
    'liver_function_score': 'Wynik czynności wątroby',
    'alpha_fetoprotein_level': 'Poziom AFP'
}

plt.figure(figsize=(12, 8))

for i, feature in enumerate(continuous_features):
    plt.subplot(2, 2, i + 1)
    sns.boxplot(y=df[feature], color='lightcoral')
    plt.title(tit[feature], fontsize=12)
    plt.ylabel(y_labels[feature], fontsize=10)

plt.subplots_adjust(top=0.9)
plt.suptitle('Wartości odstające dla zmiennych ciągłych', fontsize=16)
plt.show()

# Rozkład pacjentów chorych i zdrowych
counts = df['liver_cancer'].value_counts()
print(counts)

counts_plot = counts.rename(index={0: 'Brak raka', 1: 'Rak wątroby'})

counts_plot.plot(kind='bar', color='tan')
plt.xlabel("Wynik")
plt.ylabel("Ilość")
plt.title("Występowanie raka wątroby w badanej populacji")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Rozkład płci vs wystąpienie raka
gender_counts = df.groupby(["liver_cancer", "gender"]).size().unstack()

gender_counts = gender_counts.rename(columns={'Female': 'Kobiety', 'Male': 'Mężczyźni'})
gender_counts = gender_counts.rename(index={0: 'Brak raka', 1: 'Rak wątroby'})

gender_counts.plot(kind='bar', stacked=True, color=['tan', 'bisque'])
plt.xlabel("Wynik")
plt.ylabel("Ilość")
plt.title("Płeć a występowanie raka wątroby")
plt.xticks(rotation=45)
plt.tight_layout()
plt.legend()
plt.show()

# Identyfikacja kolumn kategorycznych i ciągłych
cat_columns = df.select_dtypes(include=['object']).columns.tolist()
num_columns = df.select_dtypes(include=[np.number]).columns.tolist()

# Kolumny kategoryczne oprócz liver_cancer
cat_columns_to_encode = [col for col in cat_columns if col != 'liver_cancer']

# Wykresy kołowe dla zmiennych kategorycznych
other_cat_columns = [
    'gender',
    'alcohol_consumption',
    'smoking_status',
    'hepatitis_b',
    'hepatitis_c',
    'cirrhosis_history',
    'family_history_cancer',
    'physical_activity_level',
    'diabetes'
]

translations = {
    'Female': 'Kobieta',
    'Male': 'Mężczyzna',
    'Never': 'Nigdy',
    'Occasional': 'Okazjonalnie',
    'Regular': 'Regularnie',
    'Former': 'Były palacz',
    'Current': 'Pali obecnie',
    0: 'Nie',
    1: 'Tak',
    'Low': 'Niskie',
    'Medium': 'Średnie',
    'Moderate': 'Umiarkowane',
    'High': 'Wysokie'
}

df_display = df.copy()
for col in other_cat_columns:
    df_display[col] = df_display[col].replace(translations)

titles1 = {
    'gender': 'Struktura płci',
    'alcohol_consumption': 'Spożycie alkoholu',
    'smoking_status': 'Status palenia',
    'hepatitis_b': 'Wirusowe zapalenie wątroby B (HBV)',
    'hepatitis_c': 'Wirusowe zapalenie wątroby C (HCV)',
    'cirrhosis_history': 'Marskość wątroby w wywiadzie',
    'family_history_cancer': 'Nowotwory w rodzinie',
    'physical_activity_level': 'Poziom aktywności fizycznej',
    'diabetes': 'Wystepowanie cukrzycy'
}

def plot_pie_group(columns, fig_title):
    num_cols = 3
    num_rows = int(np.ceil(len(columns) / num_cols))
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(6*num_cols, 4*num_rows))
    axes = axes.flatten()

    for i, column in enumerate(columns):
        value_counts = df_display[column].value_counts(normalize=True) * 100
        labels = value_counts.index
        sizes = value_counts.values

        axes[i].pie(
            sizes,
            labels=labels,
            autopct='%1.1f%%',
            startangle=140,
            textprops={'fontsize': 10}
        )
        axes[i].set_title(titles1.get(column, column), fontsize=12)

    for j in range(len(columns), len(axes)):
        axes[j].axis('off')

    fig.suptitle(fig_title, fontsize=16, fontweight='bold')
    plt.tight_layout(pad=2, rect=[0, 0, 1, 0.95])
    plt.show()

plot_pie_group(other_cat_columns, "Wykresy kołowe dla zmiennych kategorycznych")

# Historgramy dla zmiennych ciągłych
warnings.filterwarnings("ignore", message="use_inf_as_na option is deprecated", category=FutureWarning)

num_columns = ['age', 'bmi', 'liver_function_score', 'alpha_fetoprotein_level']

titles2 = {
    'age': 'Rozkład wieku pacjentów',
    'bmi': 'Rozkład wskaźnika masy ciała (BMI)',
    'liver_function_score': 'Rozkład wyniku czynności wątroby',
    'alpha_fetoprotein_level': 'Rozkład poziomu AFP'
}

translations2 = {
    'age': 'Wiek',
    'bmi': 'BMI',
    'liver_function_score': 'Wynik czynności wątroby',
    'alpha_fetoprotein_level': 'Poziom AFP'
}

num_cols = 2
num_rows = int(np.ceil(len(num_columns) / num_cols))

fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 4 * num_rows))
axes = axes.flatten()

for i, column in enumerate(num_columns):
    sns.histplot(
        df_display[column],
        kde=True,
        bins=20,
        color='skyblue',
        edgecolor='black',
        ax=axes[i]
    )
    axes[i].set_title(titles2[column], fontsize=12)
    axes[i].set_xlabel(translations2[column], fontsize=10)
    axes[i].set_ylabel('Liczba przypadków', fontsize=10)
    axes[i].tick_params(axis='x', labelsize=9)
    axes[i].tick_params(axis='y', labelsize=9)
    axes[i].grid(False)

for j in range(i + 1, len(axes)):
    axes[j].axis('off')

plt.tight_layout(pad=3)
plt.suptitle('Histogramy dla zmiennych ciągłych', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# Rozkłady zmiennych ciągłych względem liver_cancer
continuous_features = [
    'age',
    'bmi',
    'liver_function_score',
    'alpha_fetoprotein_level'
]

target = 'liver_cancer'

titles3 = {
    'age': 'Rozkład wieku względem wystąpienia raka',
    'bmi': 'Rozkład BMI względem wystąpienia raka',
    'liver_function_score': 'Rozkład wyniku czynności wątroby względem wystąpienia raka',
    'alpha_fetoprotein_level': 'Rozkład poziomu AFP względem wystąpienia raka'
}

xlabels = {
    'age': 'Wiek',
    'bmi': 'BMI',
    'liver_function_score': 'Wynik czynności wątroby',
    'alpha_fetoprotein_level': 'Poziom AFP'
}

plt.figure(figsize=(12, 10))

for i, feature in enumerate(continuous_features):
    plt.subplot(2, 2, i + 1)
    sns.kdeplot(
        data=df_display[df[target] == 0][feature],
        fill=True,
        label='Brak raka'
    )
    sns.kdeplot(
        data=df_display[df[target] == 1][feature],
        fill=True,
        label='Rak wątroby'
    )

    plt.title(titles3[feature], fontsize=12)
    plt.xlabel(xlabels[feature])
    plt.ylabel("Gęstość")
    plt.legend()

plt.tight_layout()
plt.show()

# Wykresy słupkowe obrazujące zależności między zmiennymi binarnymi a zmienną celu.
binary = ["hepatitis_b", "hepatitis_c", "cirrhosis_history",
          "family_history_cancer", "diabetes"]

binary_labels = {
    "hepatitis_b": "Wirusowe zapalenie wątroby B",
    "hepatitis_c": "Wirusowe zapalenie wątroby C",
    "cirrhosis_history": "Historia marskości wątroby",
    "family_history_cancer": "Rak w rodzinie",
    "diabetes": "Cukrzyca"
}

plot_titles = {
    "hepatitis_b": "HBV a ryzyko raka wątroby",
    "hepatitis_c": "HCV a ryzyko raka wątroby",
    "cirrhosis_history": "Marskość wątroby a ryzyko raka",
    "family_history_cancer": "Wywiad rodzinny a ryzyko raka wątroby",
    "diabetes": "Cukrzyca a ryzyko raka wątroby"
}

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for ax, col in zip(axes, binary):
    sns.countplot(
        x=col,
        data=df,
        hue="liver_cancer",
        palette="pastel",
        ax=ax
    )
    ax.set_title(plot_titles[col], fontsize=12)
    ax.set_xlabel(binary_labels[col], fontsize=10)
    ax.set_ylabel("Liczba przypadków", fontsize=10)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Nie", "Tak"])

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        labels = ["Brak raka", "Rak wątroby"]  # polskie nazwy
        ax.legend(handles, labels, loc="upper right")

if len(binary) < len(axes):
    for i in range(len(binary), len(axes)):
        fig.delaxes(axes[i])

plt.tight_layout()
plt.show()

# One Hot Encoding tylko dla zmiennych kategorycznych
df_encoded = df.copy()
df_encoded = pd.get_dummies(df_encoded, columns=cat_columns_to_encode, drop_first=True)

# Skalowanie zmiennych ciągłych
scaler = RobustScaler()
df_encoded[continuous_features] = scaler.fit_transform(df_encoded[continuous_features])
print(df_encoded.head())

# Macierz korelacji
X_corr_base = df.drop(columns=['liver_cancer'])
y_corr = df['liver_cancer']

categorical_columns = [
    'gender',
    'alcohol_consumption',
    'smoking_status',
    'hepatitis_b',
    'hepatitis_c',
    'cirrhosis_history',
    'family_history_cancer',
    'physical_activity_level',
    'diabetes'
]

numeric_columns = [
    'age',
    'bmi',
    'liver_function_score',
    'alpha_fetoprotein_level'
]

X_encoded = X_corr_base.copy()
label_enc = LabelEncoder()

for col in categorical_columns:
    X_encoded[col] = label_enc.fit_transform(X_encoded[col])

X_corr = X_encoded.copy()
X_corr['liver_cancer'] = y_corr

correlation_matrix = X_corr.corr(numeric_only=True)

translations_corr = {
    'age': 'Wiek',
    'bmi': 'BMI',
    'liver_function_score': 'Wynik czynności wątroby',
    'alpha_fetoprotein_level': 'Poziom AFP',
    'gender': 'Płeć',
    'alcohol_consumption': 'Spożycie alkoholu',
    'smoking_status': 'Palenie papierosów',
    'hepatitis_b': 'HBV',
    'hepatitis_c': 'HCV',
    'cirrhosis_history': 'Historia marskości wątroby',
    'family_history_cancer': 'Rodzinna historia raka',
    'physical_activity_level': 'Aktywność fizyczna',
    'diabetes': 'Cukrzyca',
    'liver_cancer': 'Ostateczna diagnoza'
}

correlation_matrix_polish = correlation_matrix.rename(
    index=translations_corr,
    columns=translations_corr
)

fig, ax = plt.subplots(figsize=(18, 12), constrained_layout=True)

sns.heatmap(
    correlation_matrix_polish,
    annot=True,
    fmt=".2f",
    cmap='coolwarm',
    vmin=-1,
    vmax=1,
    annot_kws={"size": 7},
    ax=ax
)

ax.set_title("Macierz korelacji zmiennych", fontsize=18, fontweight='bold')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=11)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=11)

plt.show()

X = df_encoded.drop(['liver_cancer'], axis=1)
y = df_encoded['liver_cancer']
X = X.apply(pd.to_numeric, errors='coerce')
print("Liczba NaN po konwersji:")
print(X.isna().sum())

# Imputacja medianą
imputer = SimpleImputer(strategy="median")
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
print("Liczba NaN po imputacji:")
print(X.isna().sum())
print(np.isinf(X).sum())

print("\nRozkład klas liver_cancer:")
print(y.value_counts())

print("\nŚrednie wartości numeryczne dla klas liver_cancer:")
print(df.groupby('liver_cancer').mean(numeric_only=True))

print("\nKorelacje cech z liver_cancer w df_encoded:")
print(df_encoded.corr(numeric_only=True)['liver_cancer'].sort_values(ascending=False))

# Podział 80/20 z zachowaniem proporcji klas
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"\nX_train shape: {X_train.shape}")
print(f"X_test  shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test  shape: {y_test.shape}")

print("\nRozkład klas po podziale (y_test):")
print(y_test.value_counts())

# REGRESJA LOGISTYCZNA (+ SMOTE)
print("\nBalans przed SMOTE:")
print(y_train.value_counts())

sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

print("\nBalans po SMOTE:")
print(y_train_res.value_counts())

logreg = LogisticRegression(
    solver='liblinear',
    penalty='l2',
    C=1.0,
    random_state=42
)
logreg.fit(X_train_res, y_train_res)

y_pred = logreg.predict(X_test)
y_prob = logreg.predict_proba(X_test)[:, 1]

# Macierz pomyłek
plt.figure(figsize=(5, 4))
sns.heatmap(confusion_matrix(y_test, y_pred),
            annot=True,
            fmt="d",
            cmap="Blues")
plt.title("Macierz pomyłek\n(regresja logistyczna)")
plt.xlabel("Klasa przewidywana")
plt.ylabel("Klasa rzeczywista")
plt.tight_layout()
plt.show()

# Krzywa ROC
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(5, 4))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], "--", color="gray")
plt.title("Krzywa ROC\n(regresja logistyczna)")
plt.xlabel("Fałszywe alarmy")
plt.ylabel("Czułość")
plt.legend()
plt.tight_layout()
plt.show()

# Krzywa PREC–REC
prec, rec, thr = precision_recall_curve(y_test, y_prob)

plt.figure(figsize=(5, 4))
plt.plot(rec, prec)
plt.title("Krzywa Precyzja – Czułość\n(regresja logistyczna)")
plt.xlabel("Czułość")
plt.ylabel("Precyzja")
plt.tight_layout()
plt.show()

# Różne progi decyzji
thresholds = [0.3, 0.5, 0.7]
for t in thresholds:
    y_thr = (y_prob >= t).astype(int)
    print(f"\nPROG = {t}")
    print(classification_report(y_test, y_thr))

# Wpływ regularyzacji
C_values = [0.01, 0.1, 1, 10]
auc_list = []

for C in C_values:
    model = LogisticRegression(solver='liblinear', penalty='l2', C=C, random_state=42)
    model.fit(X_train_res, y_train_res)
    y_prob_C = model.predict_proba(X_test)[:, 1]
    auc_list.append(auc(*roc_curve(y_test, y_prob_C)[:2]))

plt.figure(figsize=(5, 4))
plt.plot(C_values, auc_list, marker="o")
plt.xscale("log")
plt.title("Wpływ parametru regularyzacji C na wartość AUC\n(regresja logistyczna)")
plt.xlabel("Parametr C (odwrotność siły regularyzacji)")
plt.ylabel("Pole pod krzywą ROC (AUC)")
plt.tight_layout()
plt.show()

# Raport końcowy
print("\nRAPORT KLASYFIKACJI (domyślny próg = 0.5):")
print(classification_report(y_test, y_pred))

# SIEĆ NEURONOWA (+ SMOTE)
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)
os.environ['TF_DETERMINISTIC_OPS'] = '1'

# SMOTE (balansowanie zbioru treningowego)
print("\nBalans klas przed SMOTE:")
print(y_train.value_counts())

sm = SMOTE(random_state=42)
X_train_res2, y_train_res2 = sm.fit_resample(X_train, y_train)

print("Balans klas po SMOTE:")
print(pd.Series(y_train_res2).value_counts())

# Tworzenie modelu Dense
model = models.Sequential([
    layers.Input(shape=(X_train.shape[1],)),
    layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    layers.Dropout(0.5),
    layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
    #layers.Dense(len(set(y)), activation='softmax'
])

# Kompilacja modelu
model.compile(optimizer='adam',
              loss='binary_crossentropy', #'sparse_categorical_crossentropy'
              metrics=['accuracy'])

# Trenowanie
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5,restore_best_weights=True)
history = model.fit(X_train_res2, y_train_res2, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stop], shuffle=True)

# Ocena
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}')

# Wykresy treningu
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history.history['loss'], label='Strata treningowa')
plt.plot(history.history['val_loss'], label='Strata walidacyjna')
plt.title('Strata (loss) w trakcie treningu')
plt.xlabel('Epoka')
plt.ylabel('Wartość straty')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['accuracy'], label='Dokładność treningu')
plt.plot(history.history['val_accuracy'], label='Dokładność walidacji')
plt.title('Dokładność (accuracy) w trakcie treningu')
plt.xlabel('Epoka')
plt.ylabel('Wartość dokładności')
plt.legend()
plt.show()

# Macierz pomyłek
y_pred_probs = model.predict(X_test)
y_pred2 = (y_pred_probs > 0.5).astype(int)
#y_pred = y_pred_probs.argmax(axis=1)
cm = confusion_matrix(y_test, y_pred2)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title("Macierz pomyłek\n(sieć neuronowa)")
plt.xlabel("Klasa przewidywana")
plt.ylabel("Klasa rzeczywista")
plt.show()

# Krzywa ROC
fpr2, tpr2, _ = roc_curve(y_test, y_pred_probs)
roc_auc = auc(fpr2, tpr2)

plt.figure(figsize=(5, 4))
plt.plot(fpr2, tpr2, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], "--", color="gray")
plt.title("Krzywa ROC\n(sieć neuronowa)")
plt.xlabel("Fałszywe alarmy")
plt.ylabel("Czułość")
plt.legend()
plt.tight_layout()
plt.show()

# Krzywa PREC–REC
prec2, rec2, _ = precision_recall_curve(y_test, y_pred_probs)

plt.figure(figsize=(5, 4))
plt.plot(rec2, prec2)
plt.title("Krzywa Precyzja – Czułość\n(sieć neuronowa)")
plt.xlabel("Czułość")
plt.ylabel("Precyzja")
plt.tight_layout()
plt.show()

# Różne progi decyzji
thresholds = [0.3, 0.5, 0.7]

for t in thresholds:
    y_pred_thr2 = (y_pred_probs >= t).astype(int)
    print(f"\n=== PRÓG DECYZYJNY = {t} ===")
    print(classification_report(y_test, y_pred_thr2))

# Raport końcowy
print("\nRAPORT KLASYFIKACJI (próg = 0.5):")
print(classification_report(y_test, y_pred2))

# Stworzenie nowej sieci do sprawdzenia wpływu regularyzacji
l2_values = [0.0, 0.0001, 0.001, 0.01]
auc_list = []

for l2_lambda in l2_values:

    model = tf.keras.Sequential([
        layers.Input(shape=(X_train.shape[1],)),
        layers.Dense(
            64,
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(l2_lambda)
        ),
        layers.Dropout(0.5),
        layers.Dense(
            32,
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(l2_lambda)
        ),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    model.fit(
        X_train_res2,
        y_train_res2,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=0
    )

    y_prob = model.predict(X_test).ravel()
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_list.append(auc(fpr, tpr))

# Wykres regularyzacji
plt.figure(figsize=(5, 4))
plt.plot(l2_values, auc_list, marker='o')
plt.xscale('log')
plt.title("Wpływ regularyzacji L2 na AUC\n(sieć neuronowa)")
plt.xlabel("Współczynnik regularyzacji L2 (λ)")
plt.ylabel("Pole pod krzywą ROC (AUC)")
plt.tight_layout()
plt.show()