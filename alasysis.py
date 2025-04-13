import pandas as pd
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Загрузка JSON-файла
file_path = "C:/Users/Anton/Desktop/SuurAndmed/cs.json"

with open(file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

game_rounds = data.get("gameRounds", [])
NUM_ROUNDS = 200
extracted_features = []

for rnd in game_rounds[:NUM_ROUNDS]:
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

    # Выстрелы (до 10 сек)
    for shot in rnd.get("weaponFires", []):
        if shot.get("seconds", 999) <= 10:
            if shot.get("playerSide") == "CT":
                round_features["shots_CT"] += 1
            elif shot.get("playerSide") == "T":
                round_features["shots_T"] += 1

    # Гранаты (до 10 сек)
    for nade in rnd.get("grenades", []):
        if nade.get("throwSeconds", 999) <= 10:
            side = nade.get("throwerSide")
            gtype = nade.get("grenadeType")
            if side == "CT":
                if gtype == "Flashbang":
                    round_features["flash_CT"] += 1
                elif gtype == "Smoke Grenade":
                    round_features["smoke_CT"] += 1
                elif gtype == "HE Grenade":
                    round_features["he_CT"] += 1
                elif gtype in ["Molotov", "Incendiary Grenade"]:  # ✅ ВАЖНО!
                    round_features["molotov_CT"] += 1
            elif side == "T":
                if gtype == "Flashbang":
                    round_features["flash_T"] += 1
                elif gtype == "Smoke Grenade":
                    round_features["smoke_T"] += 1
                elif gtype == "HE Grenade":
                    round_features["he_T"] += 1
                elif gtype in ["Molotov", "Incendiary Grenade"]:  # ✅ ВАЖНО!
                    round_features["molotov_T"] += 1

    # Урон (до 10 сек)
    for dmg in rnd.get("damages", []):
        if dmg.get("seconds", 999) <= 10:
            side = dmg.get("attackerSide")
            value = dmg.get("hpDamage", 0)
            hit = dmg.get("hitGroup")
            smoke = dmg.get("isThroughSmoke", False)
            if side == "CT":
                round_features["damage_CT"] += value
                if hit == "Head":
                    round_features["headshot_damage_CT"] += value
                if smoke:
                    round_features["smoke_damage_CT"] += value
            elif side == "T":
                round_features["damage_T"] += value
                if hit == "Head":
                    round_features["headshot_damage_T"] += value
                if smoke:
                    round_features["smoke_damage_T"] += value

    # Живые игроки (на первом фрейме с <= 10 сек)
    for frame in rnd.get("frames", []):
        if frame.get("seconds", 999) <= 10:
            round_features["alive_CT"] = frame.get("ct", {}).get("alivePlayers", 0)
            round_features["alive_T"] = frame.get("t", {}).get("alivePlayers", 0)
            break

    # Убийства
    for kill in rnd.get("kills", []):
        side = kill.get("attackerSide")
        if side == "CT":
            round_features["kills_CT"] += 1
        elif side == "T":
            round_features["kills_T"] += 1

    extracted_features.append(round_features)

# Создаем DataFrame
df = pd.DataFrame(extracted_features)

# Кодируем строковые признаки в числовые
df['winningSide'] = df['winningSide'].map({'CT': 1, 'T': 0})
df['buyType_CT'] = df['buyType_CT'].map({'Full Eco': 0, 'Semi Buy': 1, 'Full Buy': 2, 'Force Buy': 3, 'Unknown': -1})
df['buyType_T'] = df['buyType_T'].map({'Full Eco': 0, 'Semi Buy': 1, 'Full Buy': 2, 'Force Buy': 3, 'Unknown': -1})

# Разделяем данные на признаки (X) и целевую переменную (y)
X = df.drop(['roundNum', 'winningSide'], axis=1)  # Убираем 'roundNum' и 'winningSide'
y = df['winningSide']  # Это целевая переменная: кто выиграл раунд (CT или T)

# Разделяем на обучающую и тестовую выборки (80% на обучение, 20% на тест)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Обучаем модель RandomForest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Прогнозируем на тестовых данных
y_pred = model.predict(X_test)

# Оценка точности
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')

# Матрица ошибок
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['CT', 'T'], yticklabels=['CT', 'T'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Дополнительно: классификационный отчет
print(classification_report(y_test, y_pred))

# Важность признаков
feature_importance = model.feature_importances_
features = X.columns
feature_df = pd.DataFrame({'Feature': features, 'Importance': feature_importance})
feature_df = feature_df.sort_values(by='Importance', ascending=False)

# Отображаем важность признаков
feature_df.plot(kind='bar', x='Feature', y='Importance', legend=False, figsize=(10, 5))
plt.title('Feature Importance')
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.show()
