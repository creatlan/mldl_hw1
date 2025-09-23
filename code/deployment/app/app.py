import streamlit as st
import requests
import math

st.set_page_config(
	page_title="Diabetes Risk Assessment",
	layout="wide"
)

st.title("Diabetes Risk Assessment Tool")
st.markdown(
	"""
	This application helps assess the likelihood of diabetes based on common clinical measurements.
	It uses a trained machine learning model (logistic regression with standardized features) to
	provide a quick risk indication. The goal is to support early awareness and encourage follow-up
	with healthcare professionals. Enter the measurements below and press Predict to see the result.
	"""
)
st.markdown("---")

col1, col2 = st.columns([2, 1])

with col1:
	st.header("Patient Health Information")
	col1_1, col1_2 = st.columns(2)
	with col1_1:
		st.subheader("Basic Information")
		pregnancies = st.number_input(
			"Number of Pregnancies",
			min_value=0,
			max_value=20,
			value=0,
			help="Number of times pregnant"
		)
		st.caption("💡 Normal: 0-5 pregnancies")
		age = st.number_input(
			"Age (years)",
			min_value=21,
			max_value=100,
			value=30,
			help="Patient must be at least 21 years old"
		)
		st.caption("💡 Normal: 21-65 years")
		st.subheader("Physical Measurements")
		st.markdown("**BMI Calculator**")
		height_cm = st.number_input(
			"Height (cm)",
			min_value=100,
			max_value=250,
			value=170,
			help="Enter height in centimeters"
		)
		weight_kg = st.number_input(
			"Weight (kg)",
			min_value=30,
			max_value=200,
			value=70,
			help="Enter weight in kilograms"
		)
		
		# Calculate BMI
		if height_cm > 0:
			bmi_calculated = weight_kg / ((height_cm / 100) ** 2)
			st.metric("Calculated BMI", f"{bmi_calculated:.1f}")
			if bmi_calculated < 18.5:
				bmi_category = "Underweight"
				bmi_color = "blue"
			elif bmi_calculated < 25:
				bmi_category = "Normal weight"
				bmi_color = "green"
			elif bmi_calculated < 30:
				bmi_category = "Overweight"
				bmi_color = "orange"
			else:
				bmi_category = "Obese"
				bmi_color = "red"
			st.markdown(f"**BMI Category:** :{bmi_color}[{bmi_category}]")
			st.caption("💡 Normal BMI: 18.5-24.9")
			bmi = bmi_calculated
		else:
			bmi = st.number_input(
				"BMI (if not calculated above)",
				min_value=10.0,
				max_value=60.0,
				value=25.0,
				help="Body Mass Index"
			)
			st.caption("💡 Normal BMI: 18.5-24.9")
	with col1_2:
		st.subheader("Blood Tests & Measurements")
		glucose = st.number_input(
			"Glucose Level (mg/dL)",
			min_value=50,
			max_value=500,
			value=100,
			help="Blood glucose level after fasting"
		)
		st.caption("💡 Normal: 70-100 mg/dL (fasting)")
		blood_pressure = st.number_input(
			"Blood Pressure (mmHg)",
			min_value=50,
			max_value=200,
			value=80,
			help="Diastolic blood pressure"
		)
		st.caption("💡 Normal: 60-80 mmHg (diastolic)")
		skin_thickness = st.number_input(
			"Skin Thickness (mm)",
			min_value=0,
			max_value=100,
			value=20,
			help="Triceps skinfold thickness"
		)
		st.caption("💡 Normal: 10-30 mm")
		insulin = st.number_input(
			"Insulin Level (μU/mL)",
			min_value=0,
			max_value=1000,
			value=15,
			help="2-hour serum insulin"
		)
		st.caption("💡 Normal: 5-25 μU/mL (fasting)")
		pedigree = st.number_input(
			"Diabetes Pedigree Function",
			min_value=0.0,
			max_value=3.0,
			value=0.25,
			step=0.1,
			help="Function representing diabetes history in relatives"
		)
		st.caption("💡 Normal: 0.0-0.5")

with col2:
	st.header("Prediction")
	if st.button("Predict", type="primary", use_container_width=True):
		try:
			with st.spinner("Analyzing health data..."):
				response = requests.post(
					"http://api:8000/predict/",
					json={
						"Pregnancies": int(pregnancies),
						"Glucose": float(glucose),
						"BloodPressure": float(blood_pressure),
						"SkinThickness": float(skin_thickness),
						"Insulin": float(insulin),
						"BMI": float(bmi),
						"DiabetesPedigreeFunction": float(pedigree),
						"Age": int(age)
					}
				)
				if response.status_code == 200:
					prediction = response.json()
					result = prediction['prediction']
					if result == 1:
						st.error("HIGH DIABETES RISK")
						st.markdown(
							"""
							Recommendation:
							- Consult with a healthcare provider
							- Consider lifestyle changes (diet, exercise)
							- Monitor blood sugar levels regularly
							"""
						)
					else:
						st.success("LOW DIABETES RISK")
						st.markdown(
							"""
							Recommendation:
							- Maintain healthy lifestyle
							- Regular health checkups
							- Balanced diet and exercise
							"""
						)
				else:
					st.error("Error connecting to prediction service")
		except requests.exceptions.ConnectionError:
			st.error("API Service Not Available")
			st.markdown(
				"""
				Please ensure the API service is running:
				1. Start the API container: docker-compose up
				2. Check if API is accessible at http://localhost:8000
				"""
			)
		except Exception as e:
			st.error(f"Error: {str(e)}")
	st.markdown("---")
	st.markdown("### Health Tips")
	st.markdown(
		"""
		- Regular Exercise: 150 minutes/week
		- Healthy Diet: Limit sugar and processed foods
		- Weight Management: Maintain healthy BMI
		- Regular Checkups: Annual health screenings
		- Stress Management: Practice relaxation techniques
		"""
	)
