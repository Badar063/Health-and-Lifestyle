import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def create_fitness_tracker_data():
    """Create fitness tracker data for 50 days"""
    np.random.seed(52)
    
    # Generate 50 consecutive days
    dates = []
    start_date = datetime(2024, 1, 1)
    for i in range(50):
        date = start_date + timedelta(days=i)
        dates.append(date.strftime('%Y-%m-%d'))
    
    # Day of week for pattern analysis
    days_of_week = [(start_date + timedelta(days=i)).strftime('%A') for i in range(50)]
    
    data = []
    
    # Base activity levels
    base_steps = 8000
    base_calories = 2000
    
    for i, (date, day) in enumerate(zip(dates, days_of_week)):
        # Weekly patterns
        if day in ['Saturday', 'Sunday']:
            activity_multiplier = np.random.uniform(0.9, 1.3)
            exercise_duration = np.random.uniform(30, 90)
        else:
            activity_multiplier = np.random.uniform(0.7, 1.1)
            exercise_duration = np.random.uniform(15, 60)
        
        # Steps with weekly pattern and some progress over time
        steps = int(base_steps * activity_multiplier * (1 + i/500))
        steps = steps + np.random.randint(-1000, 1000)
        
        # Exercise types
        exercise_types = ['Running', 'Cycling', 'Strength Training', 'Yoga', 
                         'Swimming', 'Walking', 'HIIT', 'Pilates']
        exercise_type = random.choice(exercise_types)
        
        # Exercise intensity based on type
        if exercise_type in ['HIIT', 'Running', 'Strength Training']:
            intensity = np.random.uniform(0.7, 1.0)
            calories_burned = exercise_duration * 10 * intensity
        elif exercise_type in ['Cycling', 'Swimming']:
            intensity = np.random.uniform(0.6, 0.9)
            calories_burned = exercise_duration * 8 * intensity
        else:
            intensity = np.random.uniform(0.4, 0.7)
            calories_burned = exercise_duration * 5 * intensity
        
        # Heart rate data
        resting_hr = np.random.randint(58, 72)
        max_hr = np.random.randint(resting_hr + 40, resting_hr + 100)
        avg_hr = np.random.randint(resting_hr + 10, max_hr - 10)
        
        # Distance (km)
        if exercise_type == 'Running':
            distance = exercise_duration * np.random.uniform(0.15, 0.20)  # km
        elif exercise_type == 'Cycling':
            distance = exercise_duration * np.random.uniform(0.30, 0.40)  # km
        elif exercise_type == 'Walking':
            distance = exercise_duration * np.random.uniform(0.10, 0.15)  # km
        else:
            distance = 0
        
        data.append({
            'date': date,
            'day_of_week': day,
            'steps': steps,
            'exercise_type': exercise_type,
            'exercise_duration_min': round(exercise_duration, 1),
            'exercise_intensity': round(intensity, 2),
            'calories_burned': round(calories_burned),
            'resting_heart_rate': resting_hr,
            'average_heart_rate': avg_hr,
            'max_heart_rate': max_hr,
            'distance_km': round(distance, 2),
            'active_minutes': int(exercise_duration * 0.8 + np.random.randint(10, 60)),
            'vo2_max': round(np.random.uniform(35, 50) + (i/500), 1),
            'recovery_score': np.random.randint(60, 100),
            'workout_completed': 1 if exercise_duration > 20 else 0
        })
    
    return pd.DataFrame(data)

def create_sleep_quality_data():
    """Create sleep quality metrics and influencing factors"""
    np.random.seed(53)
    
    # 45 days of sleep data
    dates = []
    start_date = datetime(2024, 1, 1)
    for i in range(45):
        date = start_date + timedelta(days=i)
        dates.append(date.strftime('%Y-%m-%d'))
    
    days_of_week = [(start_date + timedelta(days=i)).strftime('%A') for i in range(45)]
    
    data = []
    
    for i, (date, day) in enumerate(zip(dates, days_of_week)):
        # Weekend vs weekday patterns
        if day in ['Friday', 'Saturday']:
            bed_time = np.random.uniform(22, 24)  # Later bedtime
            sleep_duration = np.random.uniform(7.5, 9.5)
        else:
            bed_time = np.random.uniform(21, 23)
            sleep_duration = np.random.uniform(6.5, 8.0)
        
        wake_time = bed_time + sleep_duration
        if wake_time >= 24:
            wake_time -= 24
        
        # Sleep quality factors
        caffeine_intake = np.random.randint(0, 4)  # cups of coffee/tea
        alcohol_intake = 1 if day in ['Friday', 'Saturday'] and np.random.random() > 0.7 else 0
        stress_level = np.random.randint(1, 6)  # 1-5 scale
        screen_time_before_bed = np.random.uniform(0.5, 3.0)
        
        # Exercise effect on sleep
        exercise_today = 1 if np.random.random() > 0.3 else 0
        exercise_time = np.random.uniform(0, 2) if exercise_today else 0
        
        # Calculate sleep metrics
        sleep_efficiency = np.random.uniform(75, 95)
        if caffeine_intake > 2:
            sleep_efficiency -= np.random.uniform(5, 15)
        if alcohol_intake > 0:
            sleep_efficiency -= np.random.uniform(8, 20)
        if screen_time_before_bed > 2:
            sleep_efficiency -= np.random.uniform(5, 12)
        if exercise_today and exercise_time > 1:
            sleep_efficiency += np.random.uniform(2, 8)
        
        sleep_efficiency = max(60, min(98, sleep_efficiency))
        
        # Sleep stages
        deep_sleep = sleep_duration * np.random.uniform(0.15, 0.25)
        light_sleep = sleep_duration * np.random.uniform(0.50, 0.60)
        rem_sleep = sleep_duration * np.random.uniform(0.20, 0.25)
        
        # Sleep disturbances
        disturbances = np.random.randint(0, 5)
        if stress_level > 3:
            disturbances += np.random.randint(1, 4)
        
        # Sleep score
        sleep_score = (sleep_efficiency * 0.4 + 
                      (sleep_duration/8*100) * 0.3 +
                      (100 - disturbances*5) * 0.2 +
                      (deep_sleep/sleep_duration*100) * 0.1)
        
        data.append({
            'date': date,
            'day_of_week': day,
            'bed_time_hour': round(bed_time, 2),
            'wake_time_hour': round(wake_time, 2),
            'sleep_duration_hours': round(sleep_duration, 2),
            'sleep_efficiency_percent': round(sleep_efficiency, 1),
            'deep_sleep_hours': round(deep_sleep, 2),
            'light_sleep_hours': round(light_sleep, 2),
            'rem_sleep_hours': round(rem_sleep, 2),
            'sleep_disturbances': disturbances,
            'sleep_score': round(sleep_score, 1),
            'caffeine_intake_cups': caffeine_intake,
            'alcohol_intake_drinks': alcohol_intake,
            'stress_level': stress_level,
            'screen_time_before_bed_hours': round(screen_time_before_bed, 1),
            'exercise_before_bed': exercise_today,
            'room_temperature_c': round(np.random.uniform(18, 23), 1),
            'noise_level_db': np.random.randint(30, 60)
        })
    
    return pd.DataFrame(data)

def create_dietary_habits_data():
    """Create dietary habits and health outcomes data"""
    np.random.seed(54)
    
    # 50 days of dietary data
    dates = []
    start_date = datetime(2024, 1, 1)
    for i in range(50):
        date = start_date + timedelta(days=i)
        dates.append(date.strftime('%Y-%m-%d'))
    
    data = []
    
    # Baseline health metrics
    weight = 75.0  # kg
    energy_level = 7  # 1-10 scale
    digestion_score = 8  # 1-10 scale
    
    for i, date in enumerate(dates):
        # Dietary patterns
        meal_types = ['Omnivore', 'Vegetarian', 'Mediterranean', 'High-Protein', 
                     'Low-Carb', 'Balanced', 'Plant-Based']
        meal_type = random.choice(meal_types)
        
        # Food groups consumption (servings)
        fruits_veg = np.random.uniform(3, 8)
        proteins = np.random.uniform(2, 5)
        carbs = np.random.uniform(4, 7)
        fats = np.random.uniform(2, 4)
        sugars = np.random.uniform(0.5, 3)
        
        # Adjust based on meal type
        if meal_type == 'Vegetarian' or meal_type == 'Plant-Based':
            proteins = np.random.uniform(1.5, 3)
            fruits_veg = np.random.uniform(5, 10)
        elif meal_type == 'High-Protein':
            proteins = np.random.uniform(4, 7)
            carbs = np.random.uniform(2, 4)
        elif meal_type == 'Low-Carb':
            carbs = np.random.uniform(1, 3)
            fats = np.random.uniform(3, 6)
        
        # Nutritional totals
        calories = (fruits_veg*50 + proteins*150 + carbs*100 + fats*120 + sugars*50)
        protein_g = proteins * 20
        fiber_g = fruits_veg * 5
        sugar_g = sugars * 25
        
        # Hydration
        water_intake_liters = np.random.uniform(1.5, 3.5)
        
        # Meal timing
        breakfast_skipped = 1 if np.random.random() > 0.7 else 0
        meals_per_day = np.random.randint(2, 5)
        
        # Health outcomes (affected by diet)
        weight_change = np.random.uniform(-0.2, 0.2)
        if calories > 2500:
            weight_change += np.random.uniform(0, 0.1)
        elif calories < 1800:
            weight_change -= np.random.uniform(0, 0.1)
        
        weight += weight_change
        weight = max(65, min(85, weight))
        
        # Energy and digestion affected by diet
        energy_change = 0
        digestion_change = 0
        
        if fiber_g > 30:
            digestion_change += np.random.uniform(0.5, 1)
        if sugar_g > 50:
            energy_change -= np.random.uniform(0.5, 1.5)
            digestion_change -= np.random.uniform(0.3, 0.8)
        if water_intake_liters < 2:
            energy_change -= np.random.uniform(0.3, 0.7)
        
        energy_level = max(3, min(10, energy_level + energy_change * 0.1))
        digestion_score = max(4, min(10, digestion_score + digestion_change * 0.1))
        
        # Inflammation marker (simplified)
        inflammation_score = 5 + (sugar_g/10) - (fruits_veg/2) + np.random.uniform(-1, 1)
        inflammation_score = max(1, min(10, inflammation_score))
        
        data.append({
            'date': date,
            'diet_type': meal_type,
            'calories': round(calories),
            'protein_g': round(protein_g, 1),
            'carbs_g': round(carbs * 50, 1),
            'fats_g': round(fats * 15, 1),
            'fiber_g': round(fiber_g, 1),
            'sugar_g': round(sugar_g, 1),
            'fruits_veg_servings': round(fruits_veg, 1),
            'protein_servings': round(proteins, 1),
            'carb_servings': round(carbs, 1),
            'fat_servings': round(fats, 1),
            'water_intake_liters': round(water_intake_liters, 2),
            'breakfast_skipped': breakfast_skipped,
            'meals_per_day': meals_per_day,
            'processed_food_score': np.random.randint(1, 6),  # 1-5 scale
            'weight_kg': round(weight, 1),
            'energy_level': round(energy_level, 1),
            'digestion_score': round(digestion_score, 1),
            'inflammation_score': round(inflammation_score, 1),
            'meal_satisfaction': np.random.randint(6, 10),  # 1-10 scale
            'bloating_level': np.random.randint(1, 6)  # 1-5 scale
        })
    
    return pd.DataFrame(data)

def create_medication_adherence_data():
    """Create medication adherence and effectiveness data"""
    np.random.seed(55)
    
    # 60 days of medication data
    dates = []
    start_date = datetime(2024, 1, 1)
    for i in range(60):
        date = start_date + timedelta(days=i)
        dates.append(date.strftime('%Y-%m-%d'))
    
    # Different medications
    medications = [
        {'name': 'Metformin', 'type': 'Diabetes', 'dose': '500mg', 'frequency': 'Twice daily'},
        {'name': 'Lisinopril', 'type': 'Blood Pressure', 'dose': '10mg', 'frequency': 'Once daily'},
        {'name': 'Atorvastatin', 'type': 'Cholesterol', 'dose': '20mg', 'frequency': 'Once daily'},
        {'name': 'Levothyroxine', 'type': 'Thyroid', 'dose': '50mcg', 'frequency': 'Once daily'},
        {'name': 'Sertraline', 'type': 'Mental Health', 'dose': '50mg', 'frequency': 'Once daily'}
    ]
    
    data = []
    
    # Baseline health metrics
    bp_systolic = 140
    bp_diastolic = 90
    blood_sugar = 180
    cholesterol = 220
    thyroid_level = 8.5  # TSH
    anxiety_level = 7  # 1-10 scale
    
    for i, date in enumerate(dates):
        day_of_week = (start_date + timedelta(days=i)).strftime('%A')
        
        for med in medications:
            # Determine if medication was taken
            taken = 1
            
            # Miss doses based on day and medication type
            if day_of_week == 'Sunday' and np.random.random() > 0.8:
                taken = 0
            if med['frequency'] == 'Twice daily' and np.random.random() > 0.9:
                taken = 0.5  # Partial adherence
            
            # Time of taking (variance from prescribed time in hours)
            time_variance = np.random.uniform(-2, 2)
            
            # Side effects (random occurrence)
            side_effects = np.random.randint(0, 3) if taken > 0 else 0
            
            # Effectiveness tracking
            effectiveness = np.random.randint(7, 10) if taken == 1 else np.random.randint(3, 6)
            
            # Update health metrics based on adherence
            if taken == 1:
                if med['name'] == 'Lisinopril':
                    bp_systolic -= np.random.uniform(0.05, 0.15)
                    bp_diastolic -= np.random.uniform(0.03, 0.08)
                elif med['name'] == 'Metformin':
                    blood_sugar -= np.random.uniform(0.5, 2.0)
                elif med['name'] == 'Atorvastatin':
                    cholesterol -= np.random.uniform(0.2, 0.8)
                elif med['name'] == 'Levothyroxine':
                    thyroid_level -= np.random.uniform(0.01, 0.05)
                elif med['name'] == 'Sertraline':
                    anxiety_level -= np.random.uniform(0.05, 0.15)
            
            # Add some natural fluctuation
            bp_systolic += np.random.uniform(-1, 1)
            bp_diastolic += np.random.uniform(-0.5, 0.5)
            blood_sugar += np.random.uniform(-3, 3)
            cholesterol += np.random.uniform(-2, 2)
            thyroid_level += np.random.uniform(-0.02, 0.02)
            anxiety_level += np.random.uniform(-0.1, 0.1)
            
            # Ensure bounds
            bp_systolic = max(110, min(160, bp_systolic))
            bp_diastolic = max(70, min(100, bp_diastolic))
            blood_sugar = max(80, min(250, blood_sugar))
            cholesterol = max(150, min(280, cholesterol))
            thyroid_level = max(0.5, min(10, thyroid_level))
            anxiety_level = max(1, min(10, anxiety_level))
            
            data.append({
                'date': date,
                'day_of_week': day_of_week,
                'medication_name': med['name'],
                'medication_type': med['type'],
                'dose': med['dose'],
                'prescribed_frequency': med['frequency'],
                'taken': taken,  # 1=yes, 0=no, 0.5=partial
                'time_variance_hours': round(time_variance, 1),
                'side_effects_count': side_effects,
                'effectiveness_rating': effectiveness,  # 1-10 scale
                'bp_systolic': round(bp_systolic),
                'bp_diastolic': round(bp_diastolic),
                'blood_sugar_mgdl': round(blood_sugar),
                'cholesterol_mgdl': round(cholesterol),
                'thyroid_tsh': round(thyroid_level, 2),
                'anxiety_level': round(anxiety_level, 1),
                'mood_rating': round(10 - anxiety_level + np.random.uniform(-1, 1), 1),
                'sleep_quality_night': np.random.randint(5, 9) if taken == 1 else np.random.randint(3, 6),
                'notes': random.choice(['', 'Taken with food', 'Forgot morning dose', 'On time', '']) if taken < 1 else 'Taken as prescribed'
            })
    
    return pd.DataFrame(data)

def create_lifestyle_survey_data():
    """Create survey data about lifestyle choices and wellness indicators"""
    np.random.seed(56)
    
    # Create 50 survey responses
    participant_ids = [f'P{1000+i}' for i in range(50)]
    
    # Demographic data
    ages = np.random.randint(25, 65, 50)
    genders = np.random.choice(['Male', 'Female', 'Non-binary'], 50, p=[0.48, 0.48, 0.04])
    occupations = np.random.choice(['Office Worker', 'Healthcare', 'Education', 'Technology', 
                                   'Retail', 'Construction', 'Unemployed', 'Retired'], 50)
    
    data = []
    
    for i, pid in enumerate(participant_ids):
        age = ages[i]
        gender = genders[i]
        occupation = occupations[i]
        
        # Lifestyle factors
        exercise_frequency = np.random.choice(['Daily', '3-5 times/week', '1-2 times/week', 
                                              'Rarely', 'Never'], p=[0.2, 0.3, 0.3, 0.15, 0.05])
        
        diet_quality = np.random.randint(1, 11)  # 1-10 scale
        
        sleep_hours = np.random.uniform(5, 9)
        
        stress_level = np.random.randint(1, 11)  # 1-10 scale
        if occupation in ['Healthcare', 'Construction']:
            stress_level += np.random.randint(1, 3)
        
        social_support = np.random.randint(1, 11)
        
        alcohol_consumption = np.random.choice(['None', 'Light', 'Moderate', 'Heavy'], 
                                              p=[0.3, 0.4, 0.25, 0.05])
        
        smoking_status = np.random.choice(['Never', 'Former', 'Current'], p=[0.6, 0.3, 0.1])
        
        screen_time_hours = np.random.uniform(2, 12)
        
        # Health outcomes (influenced by lifestyle)
        bmi = np.random.uniform(18, 35)
        if exercise_frequency in ['Rarely', 'Never']:
            bmi += np.random.uniform(2, 5)
        if diet_quality < 4:
            bmi += np.random.uniform(1, 3)
        
        blood_pressure_systolic = np.random.randint(110, 150)
        blood_pressure_diastolic = np.random.randint(70, 100)
        
        if exercise_frequency in ['Daily', '3-5 times/week']:
            blood_pressure_systolic -= np.random.randint(5, 15)
            blood_pressure_diastolic -= np.random.randint(3, 8)
        
        cholesterol = np.random.randint(150, 250)
        if diet_quality > 7:
            cholesterol -= np.random.randint(10, 30)
        
        mental_health_score = np.random.randint(4, 11)
        mental_health_score += (10 - stress_level) * 0.3
        mental_health_score += social_support * 0.2
        mental_health_score = max(1, min(10, mental_health_score))
        
        energy_level = np.random.randint(4, 11)
        if sleep_hours > 7:
            energy_level += np.random.randint(1, 3)
        if exercise_frequency in ['Daily', '3-5 times/week']:
            energy_level += np.random.randint(1, 3)
        
        energy_level = max(1, min(10, energy_level))
        
        chronic_conditions = []
        if bmi > 30:
            if np.random.random() > 0.7:
                chronic_conditions.append('Type 2 Diabetes')
            if np.random.random() > 0.6:
                chronic_conditions.append('Hypertension')
        if smoking_status == 'Current':
            if np.random.random() > 0.8:
                chronic_conditions.append('COPD')
        if age > 50 and np.random.random() > 0.5:
            chronic_conditions.append('Arthritis')
        
        num_chronic_conditions = len(chronic_conditions)
        
        # Wellness indicators
        life_satisfaction = np.random.randint(5, 11)
        life_satisfaction -= stress_level * 0.2
        life_satisfaction += social_support * 0.1
        life_satisfaction = max(1, min(10, life_satisfaction))
        
        productivity_score = np.random.randint(5, 11)
        if energy_level > 7:
            productivity_score += np.random.randint(1, 3)
        productivity_score = max(1, min(10, productivity_score))
        
        data.append({
            'participant_id': pid,
            'age': age,
            'gender': gender,
            'occupation': occupation,
            'exercise_frequency': exercise_frequency,
            'diet_quality': diet_quality,
            'sleep_hours_night': round(sleep_hours, 1),
            'stress_level': stress_level,
            'social_support_score': social_support,
            'alcohol_consumption': alcohol_consumption,
            'smoking_status': smoking_status,
            'screen_time_hours_day': round(screen_time_hours, 1),
            'bmi': round(bmi, 1),
            'blood_pressure_systolic': blood_pressure_systolic,
            'blood_pressure_diastolic': blood_pressure_diastolic,
            'cholesterol_mgdl': cholesterol,
            'mental_health_score': round(mental_health_score, 1),
            'energy_level': energy_level,
            'chronic_conditions': ', '.join(chronic_conditions) if chronic_conditions else 'None',
            'num_chronic_conditions': num_chronic_conditions,
            'life_satisfaction': round(life_satisfaction, 1),
            'productivity_score': productivity_score,
            'healthcare_visits_year': np.random.randint(0, 6),
            'medication_use': 'Yes' if num_chronic_conditions > 0 and np.random.random() > 0.3 else 'No',
            'preventive_screening': 'Yes' if age > 40 and np.random.random() > 0.4 else 'No'
        })
    
    return pd.DataFrame(data)

def main():
    """Create all health datasets and save to CSV files"""
    print("Creating health and lifestyle datasets...")
    
    # Create directory if it doesn't exist
    import os
    os.makedirs('data', exist_ok=True)
    
    # Create and save each dataset
    fitness_df = create_fitness_tracker_data()
    fitness_df.to_csv('data/fitness_tracker.csv', index=False)
    print(f"Created fitness_tracker.csv with {len(fitness_df)} rows")
    
    sleep_df = create_sleep_quality_data()
    sleep_df.to_csv('data/sleep_quality.csv', index=False)
    print(f"Created sleep_quality.csv with {len(sleep_df)} rows")
    
    diet_df = create_dietary_habits_data()
    diet_df.to_csv('data/dietary_habits.csv', index=False)
    print(f"Created dietary_habits.csv with {len(diet_df)} rows")
    
    medication_df = create_medication_adherence_data()
    medication_df.to_csv('data/medication_adherence.csv', index=False)
    print(f"Created medication_adherence.csv with {len(medication_df)} rows")
    
    survey_df = create_lifestyle_survey_data()
    survey_df.to_csv('data/lifestyle_survey.csv', index=False)
    print(f"Created lifestyle_survey.csv with {len(survey_df)} rows")
    
    print("\nAll datasets created successfully!")
