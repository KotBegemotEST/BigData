import pandas as pd
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

# Загрузка JSON-файла
file_path = "C:/Users/buket/OneDrive/Desktop/data/cs.json"

with open(file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

game_rounds = data.get("gameRounds", [])
NUM_ROUNDS = 200
extracted_features = []

# Преобразуем данные
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

# Преобразуем строковые значения в столбце 'reason' в числовые (например, 'CTWin' -> 1, 'TWin' -> 0, 'Timeout' -> -1)
df['reason'] = df['reason'].map({'CTWin': 1, 'TWin': 0, 'Timeout': -1, 'BombDefused': 2, 'TimeExpired': 3, 'Unknown': -1})

# Шаг 1: Первый эксперимент с полными данными (с убийствами и причиной завершения)
# Разделяем данные на признаки (X) и целевую переменную (y)
X_full = df.drop(['roundNum', 'winningSide'], axis=1)  # Убираем 'roundNum' и 'winningSide'
y_full = df['winningSide']

# Разделяем на обучающую и тестовую выборки (80% на обучение, 20% на тест)
X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(X_full, y_full, test_size=0.2, random_state=42)

# Рассчитываем веса классов для улучшения балансировки данных
class_weights = compute_class_weight('balanced', classes=np.array([0, 1]), y=y_full)
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}

# Обучаем модель RandomForest с весами классов
model_full = RandomForestClassifier(n_estimators=100, random_state=42, class_weight=class_weight_dict)
model_full.fit(X_train_full, y_train_full)

# Прогнозируем на тестовых данных
y_pred_full = model_full.predict(X_test_full)

# Оценка точности
accuracy_full = accuracy_score(y_test_full, y_pred_full)
print(f"Шаг 1: Accuracy with full features: {accuracy_full:.4f}")

# Матрица ошибок
cm_full = confusion_matrix(y_test_full, y_pred_full)
sns.heatmap(cm_full, annot=True, fmt='d', cmap='Blues', xticklabels=['CT', 'T'], yticklabels=['CT', 'T'])
plt.title("Confusion Matrix (Full Features)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Шаг 2: Убираем очевидные признаки (убийства и причина завершения раунда)
X_reduced = X_full.drop(['kills_CT', 'kills_T', 'reason'], axis=1)  # Убираем убийства и причину завершения раунда
# Разделяем на обучающую и тестовую выборки
X_train_reduced, X_test_reduced, y_train_reduced, y_test_reduced = train_test_split(X_reduced, y_full, test_size=0.2, random_state=42)

# Обучаем модель RandomForest с весами классов
model_reduced = RandomForestClassifier(n_estimators=100, random_state=42, class_weight=class_weight_dict)
model_reduced.fit(X_train_reduced, y_train_reduced)

# Прогнозируем на тестовых данных
y_pred_reduced = model_reduced.predict(X_test_reduced)

# Оценка точности
accuracy_reduced = accuracy_score(y_test_reduced, y_pred_reduced)
print(f"Шаг 2: Accuracy without kills and reason: {accuracy_reduced:.4f}")

# Матрица ошибок
cm_reduced = confusion_matrix(y_test_reduced, y_pred_reduced)
sns.heatmap(cm_reduced, annot=True, fmt='d', cmap='Blues', xticklabels=['CT', 'T'], yticklabels=['CT', 'T'])
plt.title("Confusion Matrix (Reduced Features)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Дополнительно: классификационный отчет
print(f"Шаг 2: Classification Report without kills and reason\n{classification_report(y_test_reduced, y_pred_reduced)}")

# Важность признаков
feature_importance_reduced = model_reduced.feature_importances_
features_reduced = X_reduced.columns
feature_df_reduced = pd.DataFrame({'Feature': features_reduced, 'Importance': feature_importance_reduced})
feature_df_reduced = feature_df_reduced.sort_values(by='Importance', ascending=False)

# Отображаем важность признаков
feature_df_reduced.plot(kind='bar', x='Feature', y='Importance', legend=False, figsize=(10, 5))
plt.title('Feature Importance without kills and reason')
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.show()


# Шаг 3: Кросс-валидация

# Кросс-валидация для модели с полными признаками
cv_scores_full = cross_val_score(model_full, X_full, y_full, cv=5, scoring='accuracy')
print(f"Шаг 3: Cross-validation scores with full features: {cv_scores_full}")
print(f"Среднее значение точности с полными признаками: {cv_scores_full.mean():.4f}")

# Кросс-валидация для модели с уменьшенными признаками
cv_scores_reduced = cross_val_score(model_reduced, X_reduced, y_full, cv=5, scoring='accuracy')
print(f"Шаг 3: Cross-validation scores without kills and reason: {cv_scores_reduced}")
print(f"Среднее значение точности без убийств и причины завершения: {cv_scores_reduced.mean():.4f}")



import xgboost as xgb
from xgboost import XGBClassifier

# Шаг 4: Использование XGBoost

# Кросс-валидация для модели с полными признаками (XGBoost)
model_xgb_full = XGBClassifier(random_state=42, scale_pos_weight=class_weight_dict[1]/class_weight_dict[0])
model_xgb_full.fit(X_train_full, y_train_full)

# Прогнозируем на тестовых данных
y_pred_xgb_full = model_xgb_full.predict(X_test_full)

# Оценка точности
accuracy_xgb_full = accuracy_score(y_test_full, y_pred_xgb_full)
print(f"Шаг 4: Accuracy with full features using XGBoost: {accuracy_xgb_full:.4f}")

# Матрица ошибок
cm_xgb_full = confusion_matrix(y_test_full, y_pred_xgb_full)
sns.heatmap(cm_xgb_full, annot=True, fmt='d', cmap='Blues', xticklabels=['CT', 'T'], yticklabels=['CT', 'T'])
plt.title("Confusion Matrix (XGBoost with Full Features)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Дополнительно: классификационный отчет
print(f"Шаг 4: Classification Report using XGBoost with full features\n{classification_report(y_test_full, y_pred_xgb_full)}")

# Важность признаков для XGBoost
feature_importance_xgb_full = model_xgb_full.feature_importances_
features_full = X_full.columns
feature_df_xgb_full = pd.DataFrame({'Feature': features_full, 'Importance': feature_importance_xgb_full})
feature_df_xgb_full = feature_df_xgb_full.sort_values(by='Importance', ascending=False)

# Отображаем важность признаков для XGBoost
feature_df_xgb_full.plot(kind='bar', x='Feature', y='Importance', legend=False, figsize=(10, 5))
plt.title('Feature Importance with XGBoost (Full Features)')
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.show()

# Кросс-валидация для модели с уменьшенными признаками (XGBoost)
model_xgb_reduced = XGBClassifier(random_state=42, scale_pos_weight=class_weight_dict[1]/class_weight_dict[0])
model_xgb_reduced.fit(X_train_reduced, y_train_reduced)

# Прогнозируем на тестовых данных
y_pred_xgb_reduced = model_xgb_reduced.predict(X_test_reduced)

# Оценка точности
accuracy_xgb_reduced = accuracy_score(y_test_reduced, y_pred_xgb_reduced)
print(f"Шаг 4: Accuracy without kills and reason using XGBoost: {accuracy_xgb_reduced:.4f}")

# Матрица ошибок
cm_xgb_reduced = confusion_matrix(y_test_reduced, y_pred_xgb_reduced)
sns.heatmap(cm_xgb_reduced, annot=True, fmt='d', cmap='Blues', xticklabels=['CT', 'T'], yticklabels=['CT', 'T'])
plt.title("Confusion Matrix (XGBoost without Kills and Reason)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Дополнительно: классификационный отчет
print(f"Шаг 4: Classification Report using XGBoost without kills and reason\n{classification_report(y_test_reduced, y_pred_xgb_reduced)}")

# Важность признаков для XGBoost с уменьшенными признаками
feature_importance_xgb_reduced = model_xgb_reduced.feature_importances_
features_reduced = X_reduced.columns
feature_df_xgb_reduced = pd.DataFrame({'Feature': features_reduced, 'Importance': feature_importance_xgb_reduced})
feature_df_xgb_reduced = feature_df_xgb_reduced.sort_values(by='Importance', ascending=False)

# Отображаем важность признаков для XGBoost с уменьшенными признаками
feature_df_xgb_reduced.plot(kind='bar', x='Feature', y='Importance', legend=False, figsize=(10, 5))
plt.title('Feature Importance with XGBoost (Reduced Features)')
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.show()

# Кросс-валидация для модели с уменьшенными признаками (XGBoost)
cv_scores_xgb_reduced = cross_val_score(model_xgb_reduced, X_reduced, y_full, cv=5, scoring='accuracy')
print(f"Шаг 4: Cross-validation scores without kills and reason (XGBoost): {cv_scores_xgb_reduced}")
print(f"Среднее значение точности без убийств и причины завершения с XGBoost: {cv_scores_xgb_reduced.mean():.4f}")

