
## Step 2: Health Analysis Script

**analyze_health.py**
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class HealthLifestyleAnalyzer:
    def __init__(self):
        """Initialize the analyzer and load datasets"""
        print("Loading health and lifestyle datasets...")
        self.fitness_data = pd.read_csv('data/fitness_tracker.csv')
        self.sleep_data = pd.read_csv('data/sleep_quality.csv')
        self.diet_data = pd.read_csv('data/dietary_habits.csv')
        self.medication_data = pd.read_csv('data/medication_adherence.csv')
        self.survey_data = pd.read_csv('data/lifestyle_survey.csv')
        
        # Convert date columns to datetime
        self.fitness_data['date'] = pd.to_datetime(self.fitness_data['date'])
        self.sleep_data['date'] = pd.to_datetime(self.sleep_data['date'])
        self.diet_data['date'] = pd.to_datetime(self.diet_data['date'])
        self.medication_data['date'] = pd.to_datetime(self.medication_data['date'])
        
        print("Data loaded successfully!")
    
    def analyze_fitness_patterns(self):
        """Analyze fitness tracker data for exercise patterns and progress"""
        print("\n" + "="*60)
        print("ANALYSIS 1: FITNESS TRACKER DATA - EXERCISE PATTERNS AND PROGRESS")
        print("="*60)
        
        # Calculate weekly aggregates
        self.fitness_data['week'] = self.fitness_data['date'].dt.isocalendar().week
        self.fitness_data['week_start'] = self.fitness_data['date'] - pd.to_timedelta(self.fitness_data['date'].dt.dayofweek, unit='D')
        
        weekly_summary = self.fitness_data.groupby('week_start').agg({
            'steps': 'mean',
            'exercise_duration_min': 'mean',
            'calories_burned': 'mean',
            'distance_km': 'mean',
            'exercise_intensity': 'mean',
            'vo2_max': 'mean',
            'workout_completed': 'sum'
        }).reset_index()
        
        # Calculate progress metrics
        initial_week = weekly_summary.iloc[0]
        final_week = weekly_summary.iloc[-1]
        
        progress_metrics = {
            'Steps': ((final_week['steps'] - initial_week['steps']) / initial_week['steps'] * 100),
            'Exercise Duration': ((final_week['exercise_duration_min'] - initial_week['exercise_duration_min']) / initial_week['exercise_duration_min'] * 100),
            'Calories Burned': ((final_week['calories_burned'] - initial_week['calories_burned']) / initial_week['calories_burned'] * 100),
            'VO2 Max': ((final_week['vo2_max'] - initial_week['vo2_max']) / initial_week['vo2_max'] * 100),
            'Workouts Completed': ((final_week['workout_completed'] - initial_week['workout_completed']) / initial_week['workout_completed'] * 100)
        }
        
        print("\nProgress Over Time (% Change from First to Last Week):")
        for metric, change in progress_metrics.items():
            print(f"  {metric}: {change:.1f}%")
        
        # Exercise type analysis
        exercise_counts = self.fitness_data['exercise_type'].value_counts()
        print(f"\nMost Common Exercise Types:")
        for ex_type, count in exercise_counts.head(5).items():
            percentage = (count / len(self.fitness_data)) * 100
            print(f"  {ex_type}: {count} sessions ({percentage:.1f}%)")
        
        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Fitness Tracker Analysis', fontsize=16, fontweight='bold')
        
        # 1. Weekly progress trends
        axes[0, 0].plot(weekly_summary['week_start'], weekly_summary['steps'], 
                       marker='o', label='Steps', linewidth=2)
        axes[0, 0].plot(weekly_summary['week_start'], weekly_summary['exercise_duration_min'], 
                       marker='s', label='Exercise (min)', linewidth=2)
        axes[0, 0].set_title('Weekly Activity Trends')
        axes[0, 0].set_xlabel('Week Starting')
        axes[0, 0].set_ylabel('Activity Level')
        axes[0, 0].legend()
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Exercise type distribution
        exercise_freq = self.fitness_data['exercise_type'].value_counts()
        colors = plt.cm.Set3(np.linspace(0, 1, len(exercise_freq)))
        axes[0, 1].bar(exercise_freq.index, exercise_freq.values, color=colors)
        axes[0, 1].set_title('Exercise Type Frequency')
        axes[0, 1].set_xlabel('Exercise Type')
        axes[0, 1].set_ylabel('Number of Sessions')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Day of week patterns
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_patterns = self.fitness_data.groupby('day_of_week').agg({
            'steps': 'mean',
            'exercise_duration_min': 'mean'
        }).reindex(day_order)
        
        x = np.arange(len(day_order))
        width = 0.35
        axes[1, 0].bar(x - width/2, day_patterns['steps'], width, label='Average Steps', alpha=0.8)
        axes[1, 0].bar(x + width/2, day_patterns['exercise_duration_min'], width, 
                      label='Exercise Duration (min)', alpha=0.8)
        axes[1, 0].set_title('Activity Patterns by Day of Week')
        axes[1, 0].set_xlabel('Day of Week')
        axes[1, 0].set_ylabel('Activity Level')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(day_order, rotation=45)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # 4. Correlation matrix for fitness metrics
        fitness_corr = self.fitness_data[['steps', 'exercise_duration_min', 'exercise_intensity',
                                         'calories_burned', 'distance_km', 'vo2_max', 
                                         'resting_heart_rate']].corr()
        
        im = axes[1, 1].imshow(fitness_corr, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
        axes[1, 1].set_title('Fitness Metrics Correlation Matrix')
        axes[1, 1].set_xticks(range(len(fitness_corr.columns)))
        axes[1, 1].set_yticks(range(len(fitness_corr.columns)))
        axes[1, 1].set_xticklabels([col[:15] for col in fitness_corr.columns], rotation=45, ha='right')
        axes[1, 1].set_yticklabels([col[:15] for col in fitness_corr.columns])
        
        for i in range(len(fitness_corr.columns)):
            for j in range(len(fitness_corr.columns)):
                axes[1, 1].text(j, i, f'{fitness_corr.iloc[i, j]:.2f}',
                               ha='center', va='center', color='black', fontsize=8)
        
        plt.colorbar(im, ax=axes[1, 1])
        
        plt.tight_layout()
        plt.savefig('fitness_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return weekly_summary, progress_metrics
    
    def analyze_sleep_quality(self):
        """Study sleep quality metrics and influencing factors"""
        print("\n" + "="*60)
        print("ANALYSIS 2: SLEEP QUALITY METRICS AND INFLUENCING FACTORS")
        print("="*60)
        
        # Calculate sleep statistics
        sleep_stats = {
            'Average Sleep Duration': self.sleep_data['sleep_duration_hours'].mean(),
            'Average Sleep Efficiency': self.sleep_data['sleep_efficiency_percent'].mean(),
            'Average Deep Sleep': self.sleep_data['deep_sleep_hours'].mean(),
            'Average Sleep Score': self.sleep_data['sleep_score'].mean(),
            'Average Disturbances': self.sleep_data['sleep_disturbances'].mean()
        }
        
        print("\nSleep Statistics:")
        for stat, value in sleep_stats.items():
            print(f"  {stat}: {value:.2f}")
        
        # Correlation analysis with influencing factors
        influence_factors = ['caffeine_intake_cups', 'alcohol_intake_drinks', 
                            'stress_level', 'screen_time_before_bed_hours',
                            'exercise_before_bed', 'room_temperature_c']
        
        sleep_metrics = ['sleep_duration_hours', 'sleep_efficiency_percent', 
                        'sleep_score', 'deep_sleep_hours']
        
        print("\nCorrelation of Factors with Sleep Quality:")
        for factor in influence_factors:
            for metric in sleep_metrics:
                corr = self.sleep_data[factor].corr(self.sleep_data[metric])
                if abs(corr) > 0.3:  # Only show meaningful correlations
                    print(f"  {factor} vs {metric}: {corr:.3f}")
        
        # Weekend vs weekday comparison
        self.sleep_data['is_weekend'] = self.sleep_data['day_of_week'].isin(['Saturday', 'Sunday'])
        weekend_comparison = self.sleep_data.groupby('is_weekend').agg({
            'sleep_duration_hours': 'mean',
            'sleep_efficiency_percent': 'mean',
            'sleep_score': 'mean',
            'bed_time_hour': 'mean'
        })
        
        print(f"\nWeekend vs Weekday Sleep Patterns:")
        print(f"Weekend - Bed Time: {weekend_comparison.loc[True, 'bed_time_hour']:.1f}h")
        print(f"Weekday - Bed Time: {weekend_comparison.loc[False, 'bed_time_hour']:.1f}h")
        print(f"Weekend - Sleep Duration: {weekend_comparison.loc[True, 'sleep_duration_hours']:.1f}h")
        print(f"Weekday - Sleep Duration: {weekend_comparison.loc[False, 'sleep_duration_hours']:.1f}h")
        
        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Sleep Quality Analysis', fontsize=16, fontweight='bold')
        
        # 1. Sleep patterns over time
        axes[0, 0].plot(self.sleep_data['date'], self.sleep_data['sleep_duration_hours'], 
                       label='Duration', alpha=0.7)
        axes[0, 0].plot(self.sleep_data['date'], self.sleep_data['sleep_efficiency_percent']/10, 
                       label='Efficiency/10', alpha=0.7)
        axes[0, 0].plot(self.sleep_data['date'], self.sleep_data['sleep_score']/10, 
                       label='Score/10', alpha=0.7)
        axes[0, 0].set_title('Sleep Metrics Over Time')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Metric Value')
        axes[0, 0].legend()
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Sleep stage composition
        sleep_stages = ['deep_sleep_hours', 'light_sleep_hours', 'rem_sleep_hours']
        stage_means = [self.sleep_data[stage].mean() for stage in sleep_stages]
        stage_labels = ['Deep Sleep', 'Light Sleep', 'REM Sleep']
        colors = ['#2E86AB', '#A23B72', '#F18F01']
        
        axes[0, 1].pie(stage_means, labels=stage_labels, colors=colors, autopct='%1.1f%%',
                      startangle=90)
        axes[0, 1].set_title('Average Sleep Stage Composition')
        
        # 3. Impact of caffeine on sleep
        caffeine_levels = self.sleep_data.groupby('caffeine_intake_cups').agg({
            'sleep_duration_hours': 'mean',
            'sleep_efficiency_percent': 'mean'
        }).reset_index()
        
        ax1 = axes[1, 0]
        ax2 = ax1.twinx()
        
        ax1.bar(caffeine_levels['caffeine_intake_cups'], caffeine_levels['sleep_duration_hours'],
               alpha=0.7, color='blue', label='Sleep Duration')
        ax1.set_xlabel('Caffeine Intake (cups)')
        ax1.set_ylabel('Sleep Duration (hours)', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        
        ax2.plot(caffeine_levels['caffeine_intake_cups'], caffeine_levels['sleep_efficiency_percent'],
                color='red', marker='o', linewidth=2, label='Sleep Efficiency')
        ax2.set_ylabel('Sleep Efficiency (%)', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        
        axes[1, 0].set_title('Impact of Caffeine on Sleep')
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        # 4. Multiple regression visualization
        factors = ['stress_level', 'screen_time_before_bed_hours', 'room_temperature_c']
        n_factors = len(factors)
        
        x = np.arange(n_factors)
        correlations = [self.sleep_data[factor].corr(self.sleep_data['sleep_score']) 
                       for factor in factors]
        
        colors_corr = ['red' if corr < 0 else 'green' for corr in correlations]
        axes[1, 1].bar(x, correlations, color=colors_corr, alpha=0.7)
        axes[1, 1].set_title('Correlation with Sleep Score')
        axes[1, 1].set_xlabel('Factor')
        axes[1, 1].set_ylabel('Correlation Coefficient')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(['Stress Level', 'Screen Time', 'Room Temp'], rotation=45)
        axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('sleep_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return sleep_stats, weekend_comparison
    
    def analyze_dietary_habits(self):
        """Examine dietary habits and their relationship to health outcomes"""
        print("\n" + "="*60)
        print("ANALYSIS 3: DIETARY HABITS AND HEALTH OUTCOMES")
        print("="*60)
        
        # Calculate average nutritional intake
        nutritional_averages = {
            'Calories': self.diet_data['calories'].mean(),
            'Protein (g)': self.diet_data['protein_g'].mean(),
            'Carbs (g)': self.diet_data['carbs_g'].mean(),
            'Fats (g)': self.diet_data['fats_g'].mean(),
            'Fiber (g)': self.diet_data['fiber_g'].mean(),
            'Sugar (g)': self.diet_data['sugar_g'].mean(),
            'Water (L)': self.diet_data['water_intake_liters'].mean()
        }
        
        print("\nAverage Daily Nutritional Intake:")
        for nutrient, value in nutritional_averages.items():
            print(f"  {nutrient}: {value:.1f}")
        
        # Diet type analysis
        diet_type_stats = self.diet_data.groupby('diet_type').agg({
            'calories': 'mean',
            'energy_level': 'mean',
            'weight_kg': 'mean',
            'inflammation_score': 'mean'
        }).round(1)
        
        print("\nHealth Outcomes by Diet Type:")
        print(diet_type_stats)
        
        # Correlation analysis
        nutrients = ['calories', 'protein_g', 'fiber_g', 'sugar_g', 'water_intake_liters']
        outcomes = ['energy_level', 'weight_kg', 'digestion_score', 'inflammation_score']
        
        print("\nSignificant Correlations (|r| > 0.3):")
        for nutrient in nutrients:
            for outcome in outcomes:
                corr = self.diet_data[nutrient].corr(self.diet_data[outcome])
                if abs(corr) > 0.3:
                    print(f"  {nutrient} vs {outcome}: {corr:.3f}")
        
        # Meal timing analysis
        breakfast_analysis = self.diet_data.groupby('breakfast_skipped').agg({
            'energy_level': 'mean',
            'calories': 'mean',
            'weight_kg': 'mean'
        })
        
        print(f"\nBreakfast Skipping Impact:")
        print(f"Days with breakfast: Energy={breakfast_analysis.loc[0, 'energy_level']:.1f}, "
              f"Calories={breakfast_analysis.loc[0, 'calories']:.0f}")
        print(f"Days without breakfast: Energy={breakfast_analysis.loc[1, 'energy_level']:.1f}, "
              f"Calories={breakfast_analysis.loc[1, 'calories']:.0f}")
        
        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Dietary Habits Analysis', fontsize=16, fontweight='bold')
        
        # 1. Nutritional intake over time
        nutrients_to_plot = ['calories', 'protein_g', 'fiber_g', 'sugar_g']
        for nutrient in nutrients_to_plot:
            axes[0, 0].plot(self.diet_data['date'], self.diet_data[nutrient], 
                          label=nutrient.replace('_', ' ').title(), alpha=0.7)
        axes[0, 0].set_title('Nutritional Intake Over Time')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Amount')
        axes[0, 0].legend()
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Health outcomes by diet type
        diet_types = diet_type_stats.index
        x = np.arange(len(diet_types))
        width = 0.2
        
        metrics = ['energy_level', 'weight_kg', 'inflammation_score']
        metric_labels = ['Energy Level', 'Weight (kg)', 'Inflammation Score']
        
        for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
            offset = width * (i - 1)
            axes[0, 1].bar(x + offset, diet_type_stats[metric], width, label=label, alpha=0.7)
        
        axes[0, 1].set_title('Health Outcomes by Diet Type')
        axes[0, 1].set_xlabel('Diet Type')
        axes[0, 1].set_ylabel('Score/Value')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(diet_types, rotation=45)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # 3. Correlation heatmap
        correlation_data = self.diet_data[nutrients + outcomes].corr()
        
        im = axes[1, 0].imshow(correlation_data, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
        axes[1, 0].set_title('Nutrition vs Health Outcomes Correlation')
        axes[1, 0].set_xticks(range(len(correlation_data.columns)))
        axes[1, 0].set_yticks(range(len(correlation_data.columns)))
        axes[1, 0].set_xticklabels([col[:15] for col in correlation_data.columns], rotation=45, ha='right')
        axes[1, 0].set_yticklabels([col[:15] for col in correlation_data.columns])
        
        for i in range(len(correlation_data.columns)):
            for j in range(len(correlation_data.columns)):
                axes[1, 0].text(j, i, f'{correlation_data.iloc[i, j]:.2f}',
                               ha='center', va='center', color='black', fontsize=8)
        
        plt.colorbar(im, ax=axes[1, 0])
        
        # 4. Fiber vs digestion relationship
        fiber_bins = pd.cut(self.diet_data['fiber_g'], bins=5)
        fiber_digestion = self.diet_data.groupby(fiber_bins).agg({
            'digestion_score': 'mean',
            'bloating_level': 'mean'
        }).reset_index()
        
        ax1 = axes[1, 1]
        ax2 = ax1.twinx()
        
        x_labels = [str(interval) for interval in fiber_digestion['fiber_g']]
        x_pos = np.arange(len(x_labels))
        
        ax1.bar(x_pos, fiber_digestion['digestion_score'], alpha=0.7, color='green',
               label='Digestion Score')
        ax1.set_xlabel('Fiber Intake (g)')
        ax1.set_ylabel('Digestion Score (1-10)', color='green')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(x_labels, rotation=45)
        ax1.tick_params(axis='y', labelcolor='green')
        
        ax2.plot(x_pos, fiber_digestion['bloating_level'], color='red', 
                marker='o', linewidth=2, label='Bloating Level')
        ax2.set_ylabel('Bloating Level (1-5)', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        
        axes[1, 1].set_title('Fiber Intake Impact on Digestion')
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        plt.tight_layout()
        plt.savefig('diet_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return nutritional_averages, diet_type_stats
    
    def analyze_medication_adherence(self):
        """Track medication adherence and effectiveness over time"""
        print("\n" + "="*60)
        print("ANALYSIS 4: MEDICATION ADHERENCE AND EFFECTIVENESS")
        print("="*60)
        
        # Calculate adherence rates by medication
        adherence_stats = self.medication_data.groupby('medication_name').agg({
            'taken': 'mean',
            'effectiveness_rating': 'mean',
            'side_effects_count': 'mean'
        }).round(3)
        
        adherence_stats['adherence_rate'] = adherence_stats['taken'] * 100
        
        print("\nMedication Adherence and Effectiveness:")
        print(adherence_stats[['adherence_rate', 'effectiveness_rating', 'side_effects_count']])
        
        # Health outcomes by adherence level
        self.medication_data['adherence_category'] = pd.cut(self.medication_data['taken'], 
                                                           bins=[-0.1, 0.25, 0.75, 1.1],
                                                           labels=['Low', 'Medium', 'High'])
        
        health_by_adherence = self.medication_data.groupby(['medication_type', 'adherence_category']).agg({
            'bp_systolic': 'mean',
            'blood_sugar_mgdl': 'mean',
            'cholesterol_mgdl': 'mean',
            'anxiety_level': 'mean'
        }).round(1)
        
        print("\nHealth Outcomes by Adherence Level:")
        print(health_by_adherence)
        
        # Day of week adherence patterns
        day_adherence = self.medication_data.groupby('day_of_week').agg({
            'taken': 'mean',
            'time_variance_hours': 'mean'
        }).reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 
                   'Friday', 'Saturday', 'Sunday'])
        
        print("\nAdherence by Day of Week:")
        for day, row in day_adherence.iterrows():
            print(f"  {day}: {row['taken']*100:.1f}% adherence, "
                  f"{row['time_variance_hours']:.1f}h time variance")
        
        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Medication Adherence Analysis', fontsize=16, fontweight='bold')
        
        # 1. Adherence trends over time for each medication
        medications = self.medication_data['medication_name'].unique()
        
        for med in medications[:4]:  # Plot first 4 medications
            med_data = self.medication_data[self.medication_data['medication_name'] == med]
            daily_adherence = med_data.groupby('date')['taken'].mean().reset_index()
            axes[0, 0].plot(daily_adherence['date'], daily_adherence['taken'] * 100,
                           label=med, marker='.', alpha=0.8)
        
        axes[0, 0].set_title('Medication Adherence Over Time')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Adherence Rate (%)')
        axes[0, 0].legend()
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].axhline(y=80, color='red', linestyle='--', alpha=0.5, label='Target 80%')
        
        # 2. Adherence vs effectiveness scatter
        med_effectiveness = self.medication_data.groupby('medication_name').agg({
            'taken': 'mean',
            'effectiveness_rating': 'mean'
        }).reset_index()
        
        scatter = axes[0, 1].scatter(med_effectiveness['taken'] * 100,
                                    med_effectiveness['effectiveness_rating'],
                                    s=200, alpha=0.6)
        
        for i, row in med_effectiveness.iterrows():
            axes[0, 1].annotate(row['medication_name'][:5], 
                               (row['taken'] * 100, row['effectiveness_rating']),
                               fontsize=9, ha='center')
        
        axes[0, 1].set_title('Adherence vs Effectiveness')
        axes[0, 1].set_xlabel('Adherence Rate (%)')
        axes[0, 1].set_ylabel('Effectiveness Rating (1-10)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Health outcomes improvement with adherence
        medication_types = ['Blood Pressure', 'Diabetes', 'Cholesterol']
        
        for med_type in medication_types:
            type_data = self.medication_data[self.medication_data['medication_type'] == med_type]
            adherence_health = type_data.groupby('adherence_category').agg({
                'bp_systolic': 'mean',
                'blood_sugar_mgdl': 'mean',
                'cholesterol_mgdl': 'mean'
            }).mean(axis=1)  # Average of health metrics
            
            if med_type == 'Blood Pressure':
                axes[1, 0].plot(adherence_health.index, adherence_health.values, 
                               marker='o', label=med_type, linewidth=2)
            elif med_type == 'Diabetes':
                axes[1, 0].plot(adherence_health.index, adherence_health.values, 
                               marker='s', label=med_type, linewidth=2)
            elif med_type == 'Cholesterol':
                axes[1, 0].plot(adherence_health.index, adherence_health.values, 
                               marker='^', label=med_type, linewidth=2)
        
        axes[1, 0].set_title('Health Outcomes by Adherence Level')
        axes[1, 0].set_xlabel('Adherence Category')
        axes[1, 0].set_ylabel('Average Health Metric Value')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Day of week adherence patterns
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_patterns = day_adherence.reindex(day_order)
        
        x = np.arange(len(day_order))
        width = 0.35
        
        axes[1, 1].bar(x - width/2, day_patterns['taken'] * 100, width,
                      label='Adherence Rate', alpha=0.7)
        
        ax2 = axes[1, 1].twinx()
        ax2.bar(x + width/2, day_patterns['time_variance_hours'], width,
               color='orange', label='Time Variance (hours)', alpha=0.7)
        
        axes[1, 1].set_title('Day of Week Patterns')
        axes[1, 1].set_xlabel('Day of Week')
        axes[1, 1].set_ylabel('Adherence Rate (%)', color='blue')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(day_order, rotation=45)
        axes[1, 1].tick_params(axis='y', labelcolor='blue')
        
        ax2.set_ylabel('Time Variance (hours)', color='orange')
        ax2.tick_params(axis='y', labelcolor='orange')
        
        # Combine legends
        lines1, labels1 = axes[1, 1].get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        axes[1, 1].legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        plt.tight_layout()
        plt.savefig('medication_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return adherence_stats, health_by_adherence
    
    def analyze_lifestyle_survey(self):
        """Analyze survey data about lifestyle choices and wellness indicators"""
        print("\n" + "="*60)
        print("ANALYSIS 5: LIFESTYLE SURVEY DATA AND WELLNESS INDICATORS")
        print("="*60)
        
        # Basic statistics
        print("\nParticipant Demographics:")
        print(f"  Average Age: {self.survey_data['age'].mean():.1f} years")
        print(f"  Gender Distribution:")
        for gender, count in self.survey_data['gender'].value_counts().items():
            percentage = (count / len(self.survey_data)) * 100
            print(f"    {gender}: {count} ({percentage:.1f}%)")
        
        # Lifestyle factor analysis
        lifestyle_factors = ['exercise_frequency', 'diet_quality', 'sleep_hours_night',
                            'stress_level', 'alcohol_consumption', 'smoking_status']
        
        wellness_indicators = ['mental_health_score', 'energy_level', 'life_satisfaction',
                              'productivity_score', 'num_chronic_conditions']
        
        # Calculate correlations
        print("\nSignificant Lifestyle-Wellness Correlations (|r| > 0.3):")
        for factor in lifestyle_factors:
            if factor in ['exercise_frequency', 'alcohol_consumption', 'smoking_status']:
                # For categorical variables, compute ANOVA or group means
                continue
            
            for indicator in wellness_indicators:
                if indicator != 'num_chronic_conditions':  # Exclude count variable
                    corr = self.survey_data[factor].corr(self.survey_data[indicator])
                    if abs(corr) > 0.3:
                        print(f"  {factor} vs {indicator}: {corr:.3f}")
        
        # Exercise frequency analysis
        exercise_impact = self.survey_data.groupby('exercise_frequency').agg({
            'mental_health_score': 'mean',
            'energy_level': 'mean',
            'bmi': 'mean',
            'num_chronic_conditions': 'mean'
        }).round(2)
        
        print("\nImpact of Exercise Frequency:")
        print(exercise_impact)
        
        # Cluster analysis for lifestyle patterns
        numeric_cols = ['age', 'diet_quality', 'sleep_hours_night', 'stress_level',
                       'social_support_score', 'screen_time_hours_day', 'bmi']
        
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(self.survey_data[numeric_cols])
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        self.survey_data['lifestyle_cluster'] = kmeans.fit_predict(scaled_data)
        
        cluster_profiles = self.survey_data.groupby('lifestyle_cluster').agg({
            'diet_quality': 'mean',
            'exercise_frequency': lambda x: x.mode()[0],
            'stress_level': 'mean',
            'sleep_hours_night': 'mean',
            'bmi': 'mean',
            'mental_health_score': 'mean'
        }).round(2)
        
        print("\nLifestyle Clusters Profile:")
        print(cluster_profiles)
        
        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Lifestyle Survey Analysis', fontsize=16, fontweight='bold')
        
        # 1. Exercise frequency impact
        exercise_order = ['Daily', '3-5 times/week', '1-2 times/week', 'Rarely', 'Never']
        exercise_means = []
        
        for freq in exercise_order:
            subset = self.survey_data[self.survey_data['exercise_frequency'] == freq]
            exercise_means.append({
                'Mental Health': subset['mental_health_score'].mean(),
                'Energy Level': subset['energy_level'].mean(),
                'BMI': subset['bmi'].mean()
            })
        
        mental_health = [m['Mental Health'] for m in exercise_means]
        energy = [m['Energy Level'] for m in exercise_means]
        bmi = [m['BMI'] for m in exercise_means]
        
        x = np.arange(len(exercise_order))
        width = 0.25
        
        axes[0, 0].bar(x - width, mental_health, width, label='Mental Health', alpha=0.7)
        axes[0, 0].bar(x, energy, width, label='Energy Level', alpha=0.7)
        axes[0, 0].bar(x + width, bmi, width, label='BMI', alpha=0.7)
        
        axes[0, 0].set_title('Impact of Exercise Frequency')
        axes[0, 0].set_xlabel('Exercise Frequency')
        axes[0, 0].set_ylabel('Score/Value')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(exercise_order, rotation=45)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        
        # 2. Correlation heatmap
        corr_matrix = self.survey_data[['diet_quality', 'sleep_hours_night', 
                                       'stress_level', 'social_support_score',
                                       'mental_health_score', 'energy_level',
                                       'life_satisfaction', 'bmi']].corr()
        
        im = axes[0, 1].imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
        axes[0, 1].set_title('Lifestyle-Wellness Correlation Matrix')
        axes[0, 1].set_xticks(range(len(corr_matrix.columns)))
        axes[0, 1].set_yticks(range(len(corr_matrix.columns)))
        axes[0, 1].set_xticklabels([col[:15] for col in corr_matrix.columns], rotation=45, ha='right')
        axes[0, 1].set_yticklabels([col[:15] for col in corr_matrix.columns])
        
        for i in range(len(corr_matrix.columns)):
            for j in range(len(corr_matrix.columns)):
                axes[0, 1].text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                               ha='center', va='center', color='black', fontsize=8)
        
        plt.colorbar(im, ax=axes[0, 1])
        
        # 3. Stress vs mental health scatter with lifestyle clusters
        scatter = axes[1, 0].scatter(self.survey_data['stress_level'],
                                    self.survey_data['mental_health_score'],
                                    c=self.survey_data['lifestyle_cluster'],
                                    cmap='viridis', s=50, alpha=0.7)
        
        # Add regression line
        z = np.polyfit(self.survey_data['stress_level'], 
                      self.survey_data['mental_health_score'], 1)
        p = np.poly1d(z)
        axes[1, 0].plot(self.survey_data['stress_level'], 
                       p(self.survey_data['stress_level']), 
                       "r--", alpha=0.8, label=f'Correlation: {np.corrcoef(self.survey_data["stress_level"], self.survey_data["mental_health_score"])[0,1]:.3f}')
        
        axes[1, 0].set_title('Stress Level vs Mental Health Score')
        axes[1, 0].set_xlabel('Stress Level (1-10)')
        axes[1, 0].set_ylabel('Mental Health Score (1-10)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        plt.colorbar(scatter, ax=axes[1, 0], label='Lifestyle Cluster')
        
        # 4. Multiple lifestyle factors radar chart (simplified to bar chart)
        factors = ['diet_quality', 'sleep_hours_night', 'exercise_frequency_cat']
        
        # Convert exercise to categorical score
        exercise_map = {'Daily': 5, '3-5 times/week': 4, '1-2 times/week': 3, 
                       'Rarely': 2, 'Never': 1}
        self.survey_data['exercise_frequency_cat'] = self.survey_data['exercise_frequency'].map(exercise_map)
        
        lifestyle_scores = {
            'Diet Quality': self.survey_data['diet_quality'].mean() / 10,
            'Sleep Hours': self.survey_data['sleep_hours_night'].mean() / 9,
            'Exercise Frequency': self.survey_data['exercise_frequency_cat'].mean() / 5,
            'Social Support': self.survey_data['social_support_score'].mean() / 10,
            'Low Stress': (10 - self.survey_data['stress_level'].mean()) / 10
        }
        
        factors_names = list(lifestyle_scores.keys())
        factors_values = list(lifestyle_scores.values())
        
        angles = np.linspace(0, 2 * np.pi, len(factors_names), endpoint=False).tolist()
        factors_names += factors_names[:1]
        factors_values += factors_values[:1]
        angles += angles[:1]
        
        # Create bar chart representation instead of radar for simplicity
        x_pos = np.arange(len(factors_names[:-1]))
        axes[1, 1].bar(x_pos, factors_values[:-1], color='skyblue', alpha=0.7)
        axes[1, 1].set_title('Average Lifestyle Factor Scores (Normalized)')
        axes[1, 1].set_xlabel('Lifestyle Factor')
        axes[1, 1].set_ylabel('Normalized Score (0-1)')
        axes[1, 1].set_xticks(x_pos)
        axes[1, 1].set_xticklabels(factors_names[:-1], rotation=45)
        axes[1, 1].axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Midpoint')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('lifestyle_survey_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return exercise_impact, cluster_profiles
    
    def run_all_analysis(self):
        """Run all analyses and generate comprehensive report"""
        print("\n" + "="*60)
        print("COMPREHENSIVE HEALTH & LIFESTYLE ANALYSIS")
        print("="*60)
        
        results = {}
        
        print("\n1. Analyzing Fitness Tracker Data...")
        results['fitness'] = self.analyze_fitness_patterns()
        
        print("\n2. Analyzing Sleep Quality Data...")
        results['sleep'] = self.analyze_sleep_quality()
        
        print("\n3. Analyzing Dietary Habits Data...")
        results['diet'] = self.analyze_dietary_habits()
        
        print("\n4. Analyzing Medication Adherence Data...")
        results['medication'] = self.analyze_medication_adherence()
        
        print("\n5. Analyzing Lifestyle Survey Data...")
        results['lifestyle'] = self.analyze_lifestyle_survey()
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE!")
        print("="*60)
        
        # Generate summary report
        self.generate_summary_report(results)
        
        return results
    
    def generate_summary_report(self, results):
        """Generate a summary report of all analyses"""
        report = []
        report.append("="*60)
        report.append("HEALTH & LIFESTYLE ANALYSIS SUMMARY REPORT")
        report.append("="*60)
        report.append("")
        
        # Fitness summary
        report.append("1. FITNESS TRACKER ANALYSIS")
        report.append("-"*40)
        fitness_summary = results['fitness'][1]
        for metric, change in fitness_summary.items():
            report.append(f"{metric}: {change:.1f}% change")
        report.append("")
        
        # Sleep summary
        report.append("2. SLEEP QUALITY ANALYSIS")
        report.append("-"*40)
        sleep_stats = results['sleep'][0]
        report.append(f"Average Sleep Duration: {sleep_stats['Average Sleep Duration']:.1f} hours")
        report.append(f"Average Sleep Efficiency: {sleep_stats['Average Sleep Efficiency']:.1f}%")
        report.append(f"Average Sleep Score: {sleep_stats['Average Sleep Score']:.1f}")
        report.append("")
        
        # Diet summary
        report.append("3. DIETARY HABITS ANALYSIS")
        report.append("-"*40)
        diet_averages = results['diet'][0]
        report.append(f"Average Daily Calories: {diet_averages['Calories']:.0f}")
        report.append(f"Average Daily Protein: {diet_averages['Protein (g)']:.1f}g")
        report.append(f"Average Daily Fiber: {diet_averages['Fiber (g)']:.1f}g")
        report.append("")
        
        # Medication summary
        report.append("4. MEDICATION ADHERENCE ANALYSIS")
        report.append("-"*40)
        adherence_stats = results['medication'][0]
        best_adherence = adherence_stats['adherence_rate'].idxmax()
        worst_adherence = adherence_stats['adherence_rate'].idxmin()
        report.append(f"Highest Adherence: {best_adherence} ({adherence_stats.loc[best_adherence, 'adherence_rate']:.1f}%)")
        report.append(f"Lowest Adherence: {worst_adherence} ({adherence_stats.loc[worst_adherence, 'adherence_rate']:.1f}%)")
        report.append("")
        
        # Lifestyle summary
        report.append("5. LIFESTYLE SURVEY ANALYSIS")
        report.append("-"*40)
        cluster_profiles = results['lifestyle'][1]
        report.append(f"Number of Lifestyle Clusters Identified: {len(cluster_profiles)}")
        report.append("Cluster Characteristics:")
        for cluster, row in cluster_profiles.iterrows():
            report.append(f"  Cluster {cluster}: Diet={row['diet_quality']}, Stress={row['stress_level']}, BMI={row['bmi']}")
        
        # Key recommendations
        report.append("")
        report.append("KEY RECOMMENDATIONS")
        report.append("-"*40)
        report.append("1. Increase exercise consistency for better mental health and energy")
        report.append("2. Reduce caffeine and screen time before bed for improved sleep quality")
        report.append("3. Maintain high fiber intake for better digestion and inflammation control")
        report.append("4. Focus on medication adherence, especially on weekends")
        report.append("5. Balance lifestyle factors: diet, exercise, sleep, and stress management")
        
        # Save report to file
        with open('health_analysis_summary.txt', 'w') as f:
            f.write('\n'.join(report))
        
        print("\nSummary report saved to 'health_analysis_summary.txt'")
        print("Visualizations saved as PNG files:")
        print("- fitness_analysis.png")
        print("- sleep_analysis.png")
        print("- diet_analysis.png")
        print("- medication_analysis.png")
        print("- lifestyle_survey_analysis.png")

def main():
    """Main function to run the analysis"""
    print("Initializing Health & Lifestyle Analyzer...")
    analyzer = HealthLifestyleAnalyzer()
    
    # Run all analyses
    results = analyzer.run_all_analysis()
    
    print("\n" + "="*60)
    print("All analyses completed successfully!")
    print("Check the generated files for detailed results.")
    print("="*60)

if __name__ == "__main__":
    main()
