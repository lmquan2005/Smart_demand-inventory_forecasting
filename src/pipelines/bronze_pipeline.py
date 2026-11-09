from src.ingestion.M5_loader import M5Loader

def main():
    loader = M5Loader(config={})
    dataframes = loader.run()

if __name__ == "__main__":
    main()