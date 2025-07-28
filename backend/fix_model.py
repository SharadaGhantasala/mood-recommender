import pickle
import sys
import os

# Add the current directory to Python path so it can find our class
sys.path.append(os.getcwd())

# Now import the class
from neural_cf_ensemble import NeuralCollaborativeFiltering

def fix_model():
    print("🔧 Fixing model for API use...")
    
    try:
        # Load the original model
        with open("neural_cf_model.pkl", "rb") as f:
            model_data = pickle.load(f)
        
        print("✅ Original model loaded!")
        
        # Get the neural network object
        neural_cf = model_data['neural_cf_model']
        print(f"   Model type: {type(neural_cf)}")
        
        # Extract just the trained parts we need
        api_model = {
            'keras_model': neural_cf.model,  # The actual trained neural network
            'user_encoder': neural_cf.user_encoder,  # For converting user IDs to numbers
            'item_encoder': neural_cf.item_encoder,  # For converting track IDs to numbers
            'tracks_info': model_data['tracks_info'],  # Track information
            'model_stats': model_data['model_stats']   # Model statistics
        }
        
        # Save the API-friendly version
        with open("api_model.pkl", "wb") as f:
            pickle.dump(api_model, f)
        
        print("✅ Created api_model.pkl successfully!")
        print(f"   📊 {model_data['model_stats']['n_users']} users")
        print(f"   🎵 {model_data['model_stats']['n_items']} items")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    fix_model()
