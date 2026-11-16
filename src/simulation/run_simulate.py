from src.simulation.inventory_simulator import InventorySimulator
from pyspark.sql import functions as F

def main():
    simulator = InventorySimulator(
        random_seed=42
    )

    inventory_df = simulator.run()

    print("\n===== INVENTORY SIMULATION SUMMARY =====")

    inventory_df.groupBy().agg(
        F.count("*").alias("total_records"),
        F.sum("stockout_flag").alias("stockout_records"),
        F.sum("overstock_flag").alias("overstock_records"),
        F.sum("lost_sales").alias("total_lost_sales")
    ).show()


if __name__ == "__main__":
    main()