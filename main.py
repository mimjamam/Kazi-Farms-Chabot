"""
Main entry point for Kazi Farms Chatbot
"""
from frontend.streamlit_ui import KaziFarmFrontend

def main():
    """Main function to run the Kazi Farms Chatbot application"""
    app = KaziFarmFrontend()
    app.run()

if __name__ == "__main__":
    main()