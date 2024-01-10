import pandas as pd
import statsmodels.api as sm

data = pd.read_csv('StressLevelDataset.csv')

x = data[['anxiety_level', 'self_esteem', 'mental_health_history', 'depression', 'headache', 'blood_pressure', 'sleep_quality', 'breathing_problem', 'noise_level', 'living_conditions', 'safety', 'basic_needs', 'academic_performance', 'study_load' , 'teacher_student_relationship', 'future_career_concerns', 'social_support', 'peer_pressure', 'extracurricular_activities', 'bullying']]
y = data['stress_level']


x = sm.add_constant(x)


model = sm.OLS(y, x).fit()


durbin_watson_statistic = sm.stats.durbin_watson(model.resid)


if durbin_watson_statistic < 1.5:
    interpretation = "Autokorelasi positif (residuals cenderung positif)"
elif durbin_watson_statistic > 2.5:
    interpretation = "Autokorelasi negatif (residuals cenderung negatif)"
else:
    interpretation = "Tidak ada autokorelasi yang signifikan"

print(f"Durbin-Watson Statistic: {durbin_watson_statistic}")
print(f"Interpretasi: {interpretation}")


print(model.summary())