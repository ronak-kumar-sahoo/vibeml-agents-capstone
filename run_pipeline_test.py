import os
import asyncio
from agents import VibeMLOrchestrator

async def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(base_dir, "uploads", "churn_customer_data.csv")
    output_dir = os.path.join(base_dir, "output")
    
    print("Starting VibeML AutoML Pipeline Test Run...")
    print(f"Dataset: {dataset_path}")
    print(f"Output Directory: {output_dir}")
    
    orchestrator = VibeMLOrchestrator(dataset_path, output_dir)
    
    try:
        results = await orchestrator.run_pipeline(
            target_column="churn",
            log_callback=lambda msg: print(f"[LOG] {msg}")
        )
        
        print("\n" + "="*50)
        print("✅ PIPELINE TEST SUCCESSFUL!")
        print("="*50)
        
        print("\nGenerated files:")
        files = os.listdir(output_dir)
        for f in files:
            print(f"  ✓ {f}")
        
        model_exists = "best_model.joblib" in files
        print(f"\nModel saved: {'✅' if model_exists else '❌'}")
            
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())