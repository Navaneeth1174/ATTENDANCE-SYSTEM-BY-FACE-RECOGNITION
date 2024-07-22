# ATTENDANCE-SYSTEM-BY-FACE-RECOGNITION
Developed an advanced attendance management system leveraging face recognition technology to automate and streamline attendance tracking. The system employs a K-Nearest Neighbors (KNN) classifier for face recognition, integrated with real-time video processing for accurate and efficient attendance logging.

Key Features:
Face Detection & Recognition: Utilizes OpenCV's Haar Cascade Classifier to detect and recognize faces from live video feeds. The KNN classifier, trained on a dataset of face images, accurately identifies individuals and matches them with stored profiles.
Real-Time Processing: Captures live video input from a webcam, processes face images, and performs recognition in real-time.
Attendance Logging: Records recognized identities and timestamps in CSV files, providing a comprehensive log of attendance for each day.
Data Management: Collects and stores face data and corresponding labels in pickle files, enabling easy updates and scalable management of face profiles.
Technologies Used:

Programming Languages: Python
Libraries & Tools: OpenCV, scikit-learn (for KNN classifier), NumPy, Pickle, CSV
Deployment: Local deployment using a webcam for live data capture and processing.
Outcome:
The system enhances efficiency and accuracy in attendance tracking, reducing manual effort and potential errors. It demonstrates a practical application of machine learning and computer vision techniques in a real-world scenario.
