from src.data.loader import load_data

def main():
    # Specify the path to your CSV file
    file_path = 'dataset/sales_train_evaluation.csv'
    
    # Load the data using the load_data function
    data = load_data(file_path)
    
    # Check if data was loaded successfully
    if not data.empty:
        print("Data loaded successfully:")
        print(data.head())  # Display the first few rows of the DataFrame
    else:
        print("No data loaded.")\

if __name__ == "__main__":
    main()