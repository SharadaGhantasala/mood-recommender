import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dense, Concatenate, Dropout, Flatten
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pickle
import time
import warnings
warnings.filterwarnings('ignore')

class NeuralCollaborativeFiltering:
    def __init__(self, n_users, n_items, embedding_dim=50):
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim #how many numbers we use to describe a user/item’s personality/preferences
        self.model= None
        self.user_encoder = LabelEncoder() #LabelEncoder(): converts user_123 → 34 for training
        self.item_encoder = LabelEncoder()

    def build_model(self):
        print("building a neural network:")
        user_input =Input(shape=(), name='user_id')
        item_input = Input(shape=(), name='item_id')
        
        #making embeddings
        user_embedding = Embedding(
            input_dim= self.n_users,
            output_dim=self.embedding_dim,
            name= 'user_embedding'
        )(user_input)

        item_embedding = Embedding(
            input_dim= self.n_items,
            output_dim=self.embedding_dim,
            name= 'item_embedding'
        )(item_input)

        #concatenating user and item embeddings
        user_vec = Flatten()(user_embedding)
        item_vec = Flatten()(item_embedding)
        concat = Concatenate()([user_vec, item_vec])

        #neural network layers
        x = Dense(128, activation='relu')(concat)
        x = Dropout(0.3)(x) # dropout for regularization

        x = Dense(64, activation='relu')(x)
        x = Dropout(0.3)(x) # dropout for regularization

        x = Dense(32, activation='relu')(x)
        x = Dropout(0.3)(x) # dropout for regularization

        output = Dense(1, activation= 'sigmoid', name='prediction')(x)
        self.model = Model(inputs=[user_input, item_input], outputs=output)
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['mae'])
        print(f"   Architecture: {self.n_users} users × {self.n_items} items → {self.embedding_dim}D embeddings → Neural Network → Score")
        return self.model

    def prepare_data(self, interactions_df):
        #goal is to convert interaction data into format for neural network
        #input: dataframe with common columns: user_id, track_id, rating
        #output: arrays of numbers for tensorflow
        interactions_df['user_encoded']= self.user_encoder.fit_transform(interactions_df['user_id'])
        interactions_df['item_encoded']= self.item_encoder.fit_transform(interactions_df['track_id'])

        user_ids = interactions_df['user_encoded'].values
        item_ids= interactions_df['item_encoded'].values
        ratings = interactions_df['rating'].values

        print(f"   Encoded {len(self.user_encoder.classes_)} users and {len(self.item_encoder.classes_)} items")
        return user_ids, item_ids, ratings
    
    def train(self, interactions_df, epochs=20,  batch_size=256):
        user_ids, item_ids, ratings = self.prepare_data(interactions_df)
        
        ratings_normalized = (ratings - ratings.min()) / (ratings.max() - ratings.min())
        if self.model is None:
            self.build_model()
        
        history= self.model.fit(
            [user_ids, item_ids],
            ratings_normalized,
            epochs = epochs,
            batch_size=batch_size,
            validation_split=0.2,
            verbose=1   
        )
        return history
    def predict(self, user_ids, item_ids):
        try:
            user_encoded = self.user_encoder.transform(user_ids)
            item_encoded = self.item_encoder.transform(item_ids)
            
            predictions = self.model.predict([user_encoded, item_encoded])
            return predictions.flatten()
        except ValueError:
            print("⚠️ Warning: Unseen users/items detected")
            return np.zeros(len(user_ids))
        
    def create_synthetic_interactions(self, tracks_df, n_users=500, interactions_per_user=15):
        interactions=[]
        for user_id in range(n_users):
            user_preference_bias= np.random.uniform(0.3, 0.9)
            n_interactions = np.random.poisson(interactions_per_user)
            n_interactions = max(5, min(n_interactions, len(tracks_df)))
            popularity_weights = tracks_df['popularity'].values ** user_preference_bias
            track_indices = np.random.choice(
                len(tracks_df),
                size=n_interactions,
                replace= False,
                p= popularity_weights / popularity_weights.sum()
            )
            user_tracks= tracks_df.iloc[track_indices]
            for _, track in user_tracks.iterrows():
                base_rating = track['popularity'] / 100.0
                personal_taste = np.random.normal(0, 0.2)
                user_pickiness = np.random.normal(0, 0.1)

                rating = base_rating + personal_taste +user_pickiness
                rating =  np.clip(rating, 0, 1)
                interactions.append({
                    'user_id': f'user_{user_id}',
                    'track_id': track['track_id'],
                    'rating': rating,
                    'timestamp': time.time() + np.random.randint(-86400*30, 0)  
                })
        interactions_df = pd.DataFrame(interactions)
        print(f"Created {len(interactions_df):,} realistic interactions")
        print(f"   Average rating: {interactions_df['rating'].mean():.3f}")
        print(f"   Rating distribution: {interactions_df['rating'].describe()}")
    
        return interactions_df
def main():
    print('\n Loading music data...')
    try:
        tracks_df = pd.read_csv("large_tracks_dataset.csv")
        print(f"Loaded {len(tracks_df)} tracks")
    except FileNotFoundError:
        print("large_tracks_dataset.csv not found. Run your ETL script first.")
        return
    interactions_df = NeuralCollaborativeFiltering.create_synthetic_interactions(None, tracks_df, n_users=500, interactions_per_user=15)
    train_interactions, test_interactions = train_test_split(
    interactions_df, 
    test_size=0.2, 
    random_state=42
)
    print(f"   Training set: {len(train_interactions):,} interactions")
    print(f"   Test set: {len(test_interactions):,} interactions")
    n_users = interactions_df['user_id'].nunique()
    n_items = interactions_df['track_id'].nunique()
    neural_cf = NeuralCollaborativeFiltering(n_users, n_items, embedding_dim=50)
    print("\n Training the neural network...")
    history= neural_cf.train(train_interactions, epochs=15)
    print("\n Testing the trained model...")
    test_sample = test_interactions.head(5)
    print("Sample predictions:")
    for _, row in test_sample.iterrows():
        user_id = row['user_id']
        track_id = row['track_id']
        actual_rating = row['rating']
            
            # Find the track info for display
        track_info = tracks_df[tracks_df['track_id'] == track_id].iloc[0]
            
        print(f"   {user_id} + '{track_info['track_name']}' by {track_info['artist']}")
        print(f"   Actual rating: {actual_rating:.3f}")
        try:
            prediction = neural_cf.predict([user_id], [track_id])[0]
            print(f"   Model prediction: {prediction:.3f}")
        except:
            print(f"   Model prediction: [prediction error]")
            print()
        
        # Step 6: Save the trained model
    print("💾 Saving the trained model...")
        
    model_data = {
        'neural_cf_model': neural_cf,
        'tracks_info': tracks_df[['track_id', 'track_name', 'artist', 'popularity']].to_dict('records'),
        'training_history': history.history,
        'model_stats': {
            'n_users': n_users,
            'n_items': n_items,
            'n_interactions': len(interactions_df),
            'embedding_dim': neural_cf.embedding_dim
        }
    }
        
    with open("neural_cf_model.pkl", "wb") as f:
        pickle.dump(model_data, f)
        
    print("✅ Saved neural_cf_model.pkl")
if __name__ == "__main__":
    main()