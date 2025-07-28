import pickle
import pandas as pd

def convert_model():
    print("🔄 Converting neural CF model for API use...")
    
    try:
        # Load your original model
        with open("neural_cf_model.pkl", "rb") as f:
            model_data = pickle.load(f)
        
        neural_cf = model_data['neural_cf_model']
        tracks_info = model_data['tracks_info']
        model_stats = model_data['model_stats']
        
        print("✅ Original model loaded successfully")
        
        # Extract just the parts we need
        api_model_data = {
            'keras_model': neural_cf.model,  # The actual trained neural network
            'user_encoder': neural_cf.user_encoder,  # For encoding user IDs
            'item_encoder': neural_cf.item_encoder,  # For encoding item IDs
            'tracks_info': tracks_info,
            'model_stats': model_stats,
            'model_type': 'neural_collaborative_filtering_api'
        }
        
        # Save the API-friendly version
        with open("neural_cf_api_model.pkl", "wb") as f:
            pickle.dump(api_model_data, f)
        
        print("✅ Created neural_cf_api_model.pkl for API use")
        print(f"   📊 {model_stats['n_users']} users, {model_stats['n_items']} items")
        
        return True
        
    except Exception as e:
        print(f"❌ Error converting model: {e}")
        return False

if __name__ == "__main__":
    convert_model()
