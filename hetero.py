import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
import matplotlib.pyplot as plt


data = pd.read_csv('StressLevelDataset.csv')

x = data[['anxiety_level', 'self_esteem', 'mental_health_history', 'depression', 'headache', 'blood_pressure', 'sleep_quality', 'breathing_problem', 'noise_level', 'living_conditions', 'safety', 'basic_needs', 'academic_performance', 'study_load' , 'teacher_student_relationship', 'future_career_concerns', 'social_support', 'peer_pressure', 'extracurricular_activities', 'bullying']]
y = data['stress_level']

x = sm.add_constant(x)


model = sm.OLS(y, x).fit()


standardized_residuals = model.get_influence().resid_studentized_internal


plt.scatter(model.predict(), standardized_residuals, alpha=0.5)
plt.title('Scatterplot Prediksi vs. Residual Standar')
plt.xlabel('Nilai Prediksi')
plt.ylabel('Residual Standar')


bp_test_statistic, bp_p_value, _, _ = het_breuschpagan(standardized_residuals, x)
print(f"Breusch-Pagan Test Statistic: {bp_test_statistic}")
print(f"P-value (Breusch-Pagan Test): {bp_p_value}")

plt.show()