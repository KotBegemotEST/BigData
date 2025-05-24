import pandas as pd
import random
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


def load_json_files(paths):
    rounds = []
    for path in paths:
        with open(path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            rounds.extend(data.get("gameRounds", []))
    return rounds

def extract_features(game_rounds, seconds=30):
    extracted_features = []
    for rnd in game_rounds:
        round_features = {
            "roundNum": rnd.get("roundNum"),
            "winningSide": rnd.get("winningSide"),
            "shots_CT": 0,
            "shots_T": 0,
            "flash_CT": 0,
            "flash_T": 0,
            "smoke_CT": 0,
            "smoke_T": 0,
            "he_CT": 0,
            "he_T": 0,
            "molotov_CT": 0,
            "molotov_T": 0,
            "damage_CT": 0,
            "damage_T": 0,
            "headshot_damage_CT": 0,
            "headshot_damage_T": 0,
            "smoke_damage_CT": 0,
            "smoke_damage_T": 0,
            "alive_CT": 0,
            "alive_T": 0,
            "kills_CT": 0,
            "kills_T": 0,
            "buyType_CT": rnd.get("ctBuyType", "Unknown"),
            "buyType_T": rnd.get("tBuyType", "Unknown"),
            "spend_CT": rnd.get("ctRoundSpendMoney", 0),
            "spend_T": rnd.get("tRoundSpendMoney", 0),
            "reason": rnd.get("roundEndReason")
        }

        for shot in rnd.get("weaponFires", []):
            if shot.get("seconds", 999) <= seconds:
                if shot.get("playerSide") == "CT":
                    round_features["shots_CT"] += 1
                elif shot.get("playerSide") == "T":
                    round_features["shots_T"] += 1

        for nade in rnd.get("grenades", []):
            if nade.get("throwSeconds", 999) <= seconds:
                side = nade.get("throwerSide")
                gtype = nade.get("grenadeType")
                if side == "CT":
                    if gtype == "Flashbang": round_features["flash_CT"] += 1
                    elif gtype == "Smoke Grenade": round_features["smoke_CT"] += 1
                    elif gtype == "HE Grenade": round_features["he_CT"] += 1
                    elif gtype in ["Molotov", "Incendiary Grenade"]: round_features["molotov_CT"] += 1
                elif side == "T":
                    if gtype == "Flashbang": round_features["flash_T"] += 1
                    elif gtype == "Smoke Grenade": round_features["smoke_T"] += 1
                    elif gtype == "HE Grenade": round_features["he_T"] += 1
                    elif gtype in ["Molotov", "Incendiary Grenade"]: round_features["molotov_T"] += 1

        for dmg in rnd.get("damages", []):
            if dmg.get("seconds", 999) <= seconds:
                side = dmg.get("attackerSide")
                value = dmg.get("hpDamage", 0)
                hit = dmg.get("hitGroup")
                smoke = dmg.get("isThroughSmoke", False)
                if side == "CT":
                    round_features["damage_CT"] += value
                    if hit == "Head": round_features["headshot_damage_CT"] += value
                    if smoke: round_features["smoke_damage_CT"] += value
                elif side == "T":
                    round_features["damage_T"] += value
                    if hit == "Head": round_features["headshot_damage_T"] += value
                    if smoke: round_features["smoke_damage_T"] += value

        for frame in rnd.get("frames", []):
            if frame.get("seconds", 999) <= seconds:
                round_features["alive_CT"] = frame.get("ct", {}).get("alivePlayers", 0)
                round_features["alive_T"] = frame.get("t", {}).get("alivePlayers", 0)
                break

        for kill in rnd.get("kills", []):
            side = kill.get("attackerSide")
            if side == "CT": round_features["kills_CT"] += 1
            elif side == "T": round_features["kills_T"] += 1

        extracted_features.append(round_features)
    return pd.DataFrame(extracted_features)

def prepare_data(df, use_aggregates=True):
    df = df.copy()
    df['winningSide'] = df['winningSide'].map({'CT': 1, 'T': 0})
    df['buyType_CT'] = df['buyType_CT'].map({'Full Eco': 0, 'Semi Buy': 1, 'Full Buy': 2, 'Force Buy': 3, 'Unknown': -1})
    df['buyType_T'] = df['buyType_T'].map({'Full Eco': 0, 'Semi Buy': 1, 'Full Buy': 2, 'Force Buy': 3, 'Unknown': -1})
    df['reason'] = df['reason'].map({'CTWin': 1, 'TWin': 0, 'Timeout': -1, 'BombDefused': 2, 'TimeExpired': 3, 'Unknown': -1})

    if use_aggregates:
        df['avg_damage_CT_last3'] = df['damage_CT'].rolling(window=3, min_periods=1).mean().shift(1)
        df['avg_damage_T_last3'] = df['damage_T'].rolling(window=3, min_periods=1).mean().shift(1)
        df['avg_shots_CT_last3'] = df['shots_CT'].rolling(window=3, min_periods=1).mean().shift(1)
        df['avg_shots_T_last3'] = df['shots_T'].rolling(window=3, min_periods=1).mean().shift(1)
        df.fillna(0, inplace=True)
        print(df_all.isnull().sum().sort_values(ascending=False).head(10))
    return df

def run_experiment(df, drop_features=None, label=''):
    drop_features = drop_features if drop_features else []
    X = df.drop(['roundNum', 'winningSide'] + drop_features, axis=1)
    y = df['winningSide']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    class_weights = compute_class_weight('balanced', classes=np.array([0, 1]), y=y)
    class_weight_dict = {0: class_weights[0], 1: class_weights[1]}

    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight=class_weight_dict)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    print(f"{label} Accuracy: {accuracy:.4f}")
    print(f"{label} Cross-validation scores: {cv_scores}")
    print(f"{label} Mean CV accuracy: {cv_scores.mean():.4f}")

    return accuracy, cv_scores.mean()

all_json_paths = sorted(Path(r"C:\data").glob("cs*.json"))
paths_single = [str(all_json_paths[0])] if all_json_paths else []
paths_all = [str(p) for p in all_json_paths]



rounds_single = load_json_files(paths_single)
df_single = extract_features(rounds_single)

rounds_all = load_json_files(paths_all)
df_all = extract_features(rounds_all)
print("\n📁raundi number iga failis:")
for path in paths_all:
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        round_count = len(data.get("gameRounds", []))
        print(f"{Path(path).name}: {round_count} раундов")

# Samm 0.1: Üldine teave andmete kohta
print("\n📊 Andmestiku teave:")
print(f"Kokku raunde: {len(df_all)}")
print(f"Tunnuste arv: {df_all.shape[1]}")
print(f"Andmeraami mõõtmed: {df_all.shape}")
print("Veergude näited:", list(df_all.columns[:10]), "...")
print(df_all.describe(include='all').transpose().iloc[:10])

print("\n  Samm  1")
df_step1 = prepare_data(df_single, use_aggregates=False)
run_experiment(df_step1, label="Samm  1 (kõik tunnused)")

print("\n Samm 2")
run_experiment(df_step1, drop_features=['kills_CT', 'kills_T', 'reason'], label="Samm 2 (ilma tapmiste ja põhjusteta)")

print("\nSamm 3")
df_step3 = prepare_data(df_single, use_aggregates=True)
run_experiment(df_step3, drop_features=['kills_CT', 'kills_T', 'reason'], label="Samm 3 (agregaadid, ilma tapmiste ja põhjusteta)")

rounds_all = load_json_files(paths_all)
df_all = extract_features(rounds_all)

print("\n Samm 4")
df_step4 = prepare_data(df_all, use_aggregates=False)
run_experiment(df_step4, label="Samm 4 (kõik andmed, kõik tunnused)")

print("\n Samm 5")
run_experiment(df_step4, drop_features=['kills_CT', 'kills_T', 'reason'], label="Samm 5 (kõik andmed, ilma tapmiste ja põhjusteta)")

print("\n Samm 6")
df_step6 = prepare_data(df_all, use_aggregates=True)
run_experiment(df_step6, drop_features=['kills_CT', 'kills_T', 'reason'], label="Samm 6 (kõik andmed, agregaadid, ilma tapmiste ja põhjusteta)")
def run_feature_selection(df, drop_features=None, label=''):
    drop_features = drop_features if drop_features else []
    X = df.drop(['roundNum', 'winningSide'] + drop_features, axis=1)
    y = df['winningSide']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    class_weights = compute_class_weight('balanced', classes=np.array([0, 1]), y=y)
    class_weight_dict = {0: class_weights[0], 1: class_weights[1]}

    model_initial = RandomForestClassifier(n_estimators=100, random_state=42, class_weight=class_weight_dict)
    model_initial.fit(X_train, y_train)

    feature_importances = model_initial.feature_importances_
    features = X.columns
    feature_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
    feature_df_sorted = feature_df.sort_values(by='Importance', ascending=False)
    top_features = feature_df_sorted.head(10)['Feature'].tolist()

    print(f"\nTop 10 kõige olulisemat tunnust:\n{feature_df_sorted.head(10)}\n")

    X_top = X[top_features]
    X_train_top, X_test_top, y_train_top, y_test_top = train_test_split(X_top, y, test_size=0.2, random_state=42)

    model_top = RandomForestClassifier(n_estimators=100, random_state=42, class_weight=class_weight_dict)
    model_top.fit(X_train_top, y_train_top)

    y_pred_top = model_top.predict(X_test_top)

    accuracy_top = accuracy_score(y_test_top, y_pred_top)
    cv_scores_top = cross_val_score(model_top, X_top, y, cv=5, scoring='accuracy')

    print(f"{label} Accuracy: {accuracy_top:.4f}")
    print(f"{label} Cross-validation scores: {cv_scores_top}")
    print(f"{label} Mean CV accuracy: {cv_scores_top.mean():.4f}")

    return accuracy_top, cv_scores_top.mean(), feature_df_sorted

print("\n=== Samm 7 ===")
accuracy_top, mean_cv_top, top_feature_df = run_feature_selection(df_step6, drop_features=['kills_CT', 'kills_T', 'reason'], label="Шаг 7 (Top-10 признаков, агрегаты)")


from sklearn.feature_extraction.text import CountVectorizer


# Step 7.5 — round_type prediction using numerical features
print("\n=== Шаг 7.5 – Round Type Prediction Using Numerical Features ===")

def label_round_type(row):
    if row['shots_T'] > 60 and row['smoke_T'] > 2:
        return "Fast Push"
    elif row['smoke_T'] >= 3 and row['flash_T'] >= 3:
        return "Slow Default"
    elif row['smoke_T'] >= 2 and row['flash_T'] == 0:
        return "Fake"
    elif row['alive_T'] == 1 and row['kills_T'] >= 2:
        return "Clutch"
    elif row['spend_T'] < 2000:
        return "Save"
    else:
        return "Default"

df_step6 = df_step6.copy()
df_step6["round_type"] = df_step6.apply(label_round_type, axis=1)

top_10_features = top_feature_df.head(10)["Feature"].tolist()
X = df_step6[top_10_features]
y = df_step6["round_type"]

# Random Forest
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Step 7.5 Accuracy: {accuracy:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred, labels=model.classes_)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
plt.title("Step 7.5 – Round Type Prediction Using Numerical Features")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

print("\n Samm 8: Raundi tüübi klassifitseerimine (BOW-laadne)")

def create_text_description(row):
    return (
        f"shots_CT:{row['shots_CT']} shots_T:{row['shots_T']} "
        f"flash_CT:{row['flash_CT']} flash_T:{row['flash_T']} "
        f"smoke_CT:{row['smoke_CT']} smoke_T:{row['smoke_T']} "
        f"molotov_CT:{row['molotov_CT']} molotov_T:{row['molotov_T']} "
        f"alive_CT:{row['alive_CT']} alive_T:{row['alive_T']} "
        f"spend_CT:{row['spend_CT']} spend_T:{row['spend_T']} "
        f"buy_CT:{row['buyType_CT']} buy_T:{row['buyType_T']}"
    )


def label_round_type(row):
    if row['alive_T'] == 1 and row['kills_T'] >= 2 and row['damage_T'] > 150:
        return "Clutch"
    elif row['spend_T'] < 1500 and row['shots_T'] < 10 and row['damage_T'] < 50:
        return "Save"
    elif row['smoke_T'] >= 3 and row['flash_T'] <= 1 and row['damage_T'] < 80:
        return "Fake"
    elif row['smoke_T'] >= 2 and row['flash_T'] >= 3 and row['alive_T'] >= 3 and row['damage_T'] < 150:
        return "Slow Default"
    elif row['shots_T'] > 40 and row['damage_T'] > 150 and row['alive_CT'] <= 2:
        return "Fast Push"
    else:
        return "Default"



df_bow = df_step6.copy()
df_bow["round_type"] = df_bow.apply(label_round_type, axis=1)
df_bow["text"] = df_bow.apply(create_text_description, axis=1)

texts = df_bow["text"]
labels = df_bow["round_type"]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)
y = labels


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model_bow = RandomForestClassifier(n_estimators=100, random_state=42)
model_bow.fit(X_train, y_train)

y_pred = model_bow.predict(X_test)


print("Raunditüüpide klassifitseerimine (BOW-laadne):")
print("Täpsus (Accuracy):", accuracy_score(y_test, y_pred))
print("Klasside aruanne:\n", classification_report(y_test, y_pred))
print("Segadusmaatriks:\n", confusion_matrix(y_test, y_pred))

print("\n Samm 9: Klassifitseerimine TF-IDF abil")


texts = df_bow["text"]
labels = df_bow["round_type"]


tfidf_vectorizer = TfidfVectorizer()
X_tfidf = tfidf_vectorizer.fit_transform(texts)
y_tfidf = labels

X_train_tfidf, X_test_tfidf, y_train_tfidf, y_test_tfidf = train_test_split(X_tfidf, y_tfidf, test_size=0.2, random_state=42)

model_tfidf = RandomForestClassifier(n_estimators=100, random_state=42)
model_tfidf.fit(X_train_tfidf, y_train_tfidf)
y_pred_tfidf = model_tfidf.predict(X_test_tfidf)


print("Täpsus (TF-IDF):", accuracy_score(y_test_tfidf, y_pred_tfidf))
print("Klasside aruanne:\n", classification_report(y_test_tfidf, y_pred_tfidf))
print("Segadusmaatriks:\n", confusion_matrix(y_test_tfidf, y_pred_tfidf))


steps = [
    "Step 1\nSingle file\nAll features",
    "Step 2\nSingle file\nNo kills/reason",
    "Step 3\nSingle file\n+Aggregates",
    "Step 4\nAll files\nAll features",
    "Step 5\nAll files\nNo kills/reason",
    "Step 6\nAll files\n+Aggregates",
    "Step 7\nAll files\nTop 10 features"
]
accuracies = [1.0, 0.1667, 0.3333, 0.9815, 0.7963, 0.7778, 0.8333]
cv_means = [0.9333, 0.5333, 0.5, 0.9665, 0.7509, 0.7697, 0.7804]

# Create plot
plt.figure(figsize=(12, 6))
plt.plot(steps, accuracies, marker='o', label='Accuracy')
plt.plot(steps, cv_means, marker='s', label='Mean Cross-Validation Accuracy')
plt.xticks(rotation=30, ha='right')
plt.ylabel('Score')
plt.title('Model Performance Across Steps')
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.show()


plt.figure(figsize=(6, 4))
sns.countplot(x='winningSide', data=df_all, palette='pastel')
plt.title("Round Outcomes by Side")
plt.xlabel("Winning Side")
plt.ylabel("Number of Rounds")
plt.tight_layout()
plt.show()


plt.figure(figsize=(8, 4))
sns.boxplot(data=df_all[['damage_CT', 'damage_T']], palette='muted')
plt.title("Damage Distribution (CT vs T)")
plt.ylabel("HP Damage")
plt.tight_layout()
plt.show()


plt.figure(figsize=(14, 10))
corr = df_all.corr(numeric_only=True)
sns.heatmap(corr, cmap='coolwarm', annot=False, fmt=".2f")
plt.title("Correlation Matrix of Numerical Features")
plt.tight_layout()
plt.show()


def categorize_numeric(value, bins, labels):
    for threshold, label in zip(bins, labels):
        if value <= threshold:
            return label
    return labels[-1]



synonyms = {
    "T side opened fire": [
        "T side opened fire",
        "T players started shooting",
        "T side initiated an attack",
        "T side engaged with bullets"
    ],
    "CT side responded with": [
        "CT side responded with",
        "CT answered with",
        "CT retaliated using",
        "CT side returned fire with"
    ],
    "used flashbangs": [
        "used flashbangs",
        "threw flash grenades",
        "deployed flashes",
        "executed flash throws"
    ],
    "deployed smoke grenades": [
        "deployed smoke grenades",
        "used smokes",
        "threw smoke bombs",
        "executed smoke cover"
    ],
    "inflicted damage": [
        "inflicted damage",
        "caused damage",
        "dealt damage",
        "delivered hits"
    ]
}

def build_semantic_text_advanced(row):
    def cat(value, bins, labels):
        for threshold, label in zip(bins, labels):
            if value <= threshold:
                return label
        return labels[-1]

    import random

    shots_T_level = cat(row["shots_T"], [10, 25, 40, 60], ["very low", "low", "moderate", "high", "very high"])
    shots_CT_level = cat(row["shots_CT"], [10, 25, 40, 60], ["very low", "low", "moderate", "high", "very high"])

    flash_T_level = cat(row["flash_T"], [0, 1, 2, 4], ["none", "few", "some", "many", "a lot"])
    flash_CT_level = cat(row["flash_CT"], [0, 1, 2, 4], ["none", "few", "some", "many", "a lot"])

    smoke_T_level = cat(row["smoke_T"], [0, 1, 2, 3], ["none", "few", "some", "many", "a lot"])
    smoke_CT_level = cat(row["smoke_CT"], [0, 1, 2, 3], ["none", "few", "some", "many", "a lot"])

    damage_T_level = cat(row["damage_T"], [50, 100, 200, 400], ["very low", "low", "moderate", "high", "very high"])
    damage_CT_level = cat(row["damage_CT"], [50, 100, 200, 400], ["very low", "low", "moderate", "high", "very high"])

    alive_T = row["alive_T"]
    alive_CT = row["alive_CT"]

    buy_T = row.get("buyType_T", -1)
    buy_CT = row.get("buyType_CT", -1)
    buy_dict = {0: "an eco", 1: "a semi-buy", 2: "a full-buy", 3: "a force-buy", -1: "an unknown buy"}

    paragraph = []

    paragraph.append(
        f"The round began with the T side initiating {random.choice(['aggressive fire', 'an opening spray', 'initial shots'])} of {shots_T_level} intensity."
    )
    paragraph.append(
        f"CT players responded with {shots_CT_level} return fire, aiming to counter the opening momentum."
    )

    if flash_T_level == "none" and smoke_T_level in ["many", "a lot"]:
        paragraph.append("Interestingly, the T side relied heavily on smokes without any flashes, possibly suggesting a fake.")
    else:
        paragraph.append(
            f"The T side used {flash_T_level} flashbangs and {smoke_T_level} smoke grenades to create space."
        )

    if flash_CT_level != "none" or smoke_CT_level != "none":
        paragraph.append(
            f"Meanwhile, the CT side used {flash_CT_level} flashbangs and {smoke_CT_level} smokes to delay the execution."
        )

    paragraph.append(
        f"The T side inflicted {damage_T_level} total damage, while CT players caused {damage_CT_level} in return."
    )

    if alive_T == 1 and row["kills_T"] >= 2:
        paragraph.append("Despite only one T player remaining, the side managed to secure multiple kills — a clutch attempt was likely.")
    elif alive_CT < alive_T:
        paragraph.append("The CTs were at a numerical disadvantage as the round progressed.")
    elif alive_T < alive_CT:
        paragraph.append("The Ts found themselves outnumbered nearing the end of the round.")


    paragraph.append(
        f"T side entered the round with {buy_dict.get(buy_T)} setup, while CTs responded with {buy_dict.get(buy_CT)}."
    )

    paragraph.append(
        random.choice([
            "The round featured dynamic utility usage and fast-paced decisions.",
            "Both sides showed clear tactical intent, resulting in a well-fought engagement.",
            "Strategic positioning and mid-round adaptation defined this round."
        ])
    )

    return " ".join(paragraph)


df_semantic = df_step6.copy()
df_semantic["round_type"] = df_semantic.apply(label_round_type, axis=1)

df_semantic["semantic_text"] = df_semantic.apply(build_semantic_text_advanced, axis=1)

print(df_semantic["semantic_text"].iloc[0])


from sklearn.feature_extraction.text import TfidfVectorizer

print("\n=== Samm 10: Semantiline TF-IDF ===")

texts_semantic = df_semantic["semantic_text"]
labels_semantic = df_semantic["round_type"]

tfidf_vectorizer = TfidfVectorizer()
X_semantic_tfidf = tfidf_vectorizer.fit_transform(texts_semantic)
y_semantic = labels_semantic

X_train_sem, X_test_sem, y_train_sem, y_test_sem = train_test_split(X_semantic_tfidf, y_semantic, test_size=0.2, random_state=42)

model_sem = RandomForestClassifier(n_estimators=100, random_state=42)
model_sem.fit(X_train_sem, y_train_sem)

y_pred_sem = model_sem.predict(X_test_sem)


print("Täpsus (semantiline TF-IDF):", accuracy_score(y_test_sem, y_pred_sem))
print("Klasside aruanne:\n", classification_report(y_test_sem, y_pred_sem))
print("Segadusmaatriks:\n", confusion_matrix(y_test_sem, y_pred_sem))

print("\n Samm 11: Ebmeding")


model_embed = SentenceTransformer('paraphrase-MiniLM-L6-v2')

X_embed = model_embed.encode(df_semantic["semantic_text"].tolist())
y_embed = df_semantic["round_type"]

X_train_emb, X_test_emb, y_train_emb, y_test_emb = train_test_split(X_embed, y_embed, test_size=0.2, random_state=42)
clf_embed = RandomForestClassifier(n_estimators=100, random_state=42)
clf_embed.fit(X_train_emb, y_train_emb)
y_pred_emb = clf_embed.predict(X_test_emb)

print("Accuracy (Sentence Embeddings):", accuracy_score(y_test_emb, y_pred_emb))
print("Aruanne:\n", classification_report(y_test_emb, y_pred_emb))


####################

# Step 12 – Logistic Regression on Sentence Embeddings


X_train_lr, X_test_lr, y_train_lr, y_test_lr = train_test_split(X_embed, y_embed, test_size=0.2, random_state=42)

clf_lr = LogisticRegression(max_iter=1000, solver='lbfgs', multi_class='multinomial')
clf_lr.fit(X_train_lr, y_train_lr)

y_pred_lr = clf_lr.predict(X_test_lr)

print("=== Step 12: Logistic Regression on Sentence Embeddings ===")
print("Accuracy:", accuracy_score(y_test_lr, y_pred_lr))
print("Classification Report:\n", classification_report(y_test_lr, y_pred_lr))
print("Confusion Matrix:\n", confusion_matrix(y_test_lr, y_pred_lr))
