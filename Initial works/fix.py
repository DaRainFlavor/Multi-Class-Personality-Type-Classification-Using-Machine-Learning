import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# Question-to-MBTI Dimension Mapping
# ============================================================================

QUESTION_MAPPING = {
    'E_I': {
        'positive': [  # High score = Extraversion
            'You regularly make new friends.',
            'You feel comfortable just walking up to someone you find interesting and striking up a conversation.',
            'You enjoy participating in group activities.',
            'You usually prefer to be around others rather than on your own.',
            'After a long and exhausting week, a lively social event is just what you need.',
            'In your social circle, you are often the one who contacts your friends and initiates activities.',
            'You feel more drawn to places with busy, bustling atmospheres than quiet, intimate places.',
        ],
        'negative': [  # High score = Introversion (reverse for E)
            'At social events, you rarely try to introduce yourself to new people and mostly talk to the ones you already know',
            'You tend to avoid drawing attention to yourself.',
            'You avoid leadership roles in group settings.',
            'You avoid making phone calls.',
            'You would love a job that requires you to work alone most of the time.',
        ],
    },
    'S_N': {
        'positive': [  # High score = Intuition
            'You spend a lot of your free time exploring various random topics that pique your interest',
            'You like books and movies that make you come up with your own interpretation of the ending.',
            'You are interested in so many things that you find it difficult to choose what to try next.',
            'You have always been fascinated by the question of what, if anything, happens after death.',
            'You enjoy going to art museums.',
            'You often spend a lot of time trying to understand views that are very different from your own.',
            'You are very intrigued by things labeled as controversial.',
        ],
        'negative': [  # High score = Sensing (reverse for N)
            'You are not too interested in discussing various interpretations and analyses of creative works.',
            'You are definitely not an artistic type of person.',
            'You become bored or lose interest when the discussion gets highly theoretical.',
            'You rarely contemplate the reasons for human existence or the meaning of life.',
            'You believe that pondering abstract philosophical questions is a waste of time.',
        ],
    },
    'T_F': {
        'positive': [  # High score = Feeling
            'Seeing other people cry can easily make you feel like you want to cry too',
            'You are very sentimental.',
            'Your happiness comes more from helping others accomplish things than your own accomplishments.',
            'You find it easy to empathize with a person whose experiences are very different from yours.',
            'Your emotions control you more than you control them.',
            'You take great care not to make people look bad, even when it is completely their fault.',
            'You know at first glance how someone is feeling.',
            'You would pass along a good opportunity if you thought someone else needed it more.',
        ],
        'negative': [  # High score = Thinking (reverse for F)
            'You are more inclined to follow your head than your heart.',
            'You think the world would be a better place if people relied more on rationality and less on their feelings.',
            'You lose patience with people who are not as efficient as you.',
            'You enjoy watching people argue.',
            'You often have a hard time understanding other people\'s feelings.',
        ],
    },
    'J_P': {
        'positive': [  # High score = Judging
            'You often make a backup plan for a backup plan.',
            'You prefer to completely finish one project before starting another.',
            'You like to use organizing tools like schedules and lists.',
            'You prefer to do your chores before allowing yourself to relax.',
            'You like to have a to-do list for each day.',
            'If your plans are interrupted, your top priority is to get back on track as soon as possible.',
            'You complete things methodically without skipping over any steps.',
        ],
        'negative': [  # High score = Perceiving (reverse for J)
            'You usually prefer just doing what you feel like at any given moment instead of planning a particular daily routine.',
            'You often end up doing things at the last possible moment.',
            'You usually postpone finalizing decisions for as long as possible.',
            'Your personal work style is closer to spontaneous bursts of energy than organized and consistent efforts.',
            'You struggle with deadlines.',
        ],
    },
    'TURBULENCE': {
        'positive': [
            'Even a small mistake can cause you to doubt your overall abilities and knowledge.',
            'You are prone to worrying that things will take a turn for the worse.',
            'Your mood can change very quickly.',
            'You are still bothered by mistakes that you made a long time ago.',
            'When someone thinks highly of you, you wonder how long it will take them to feel disappointed in you.',
            'You often feel overwhelmed.',
        ],
        'negative': [
            'You usually stay calm, even under a lot of pressure',
            'You rarely worry about whether you make a good impression on people you meet.',
            'You rarely second-guess the choices that you have made.',
            'You rarely feel insecure.',
            'You feel confident that things will work out for you.',
        ],
    }
}

def load_data(filepath, encoding='cp1252'):
    """Load the CSV dataset."""
    print("=" * 60)
    print("STEP 1: Loading Dataset")
    print("=" * 60)
    
    df = pd.read_csv(filepath, encoding=encoding)
    print(f"Loaded {len(df)} rows and {len(df.columns)} columns")
    print(f"\nPersonality distribution:")
    print(df['Personality'].value_counts())
    
    return df

def analyze_and_clean(df):
    """Analyze and clean dataset."""
    print("\n" + "=" * 60)
    print("STEP 2: Analyzing & Cleaning Data")
    print("=" * 60)
    
    question_cols = [c for c in df.columns if c not in ['Response Id', 'Personality']]
    
    print(f"\n1. Missing values: {df.isnull().sum().sum()}")
    print(f"2. Question columns: {len(question_cols)}")
    print(f"3. Value range: [{df[question_cols].min().min()}, {df[question_cols].max().max()}]")
    
    # Calculate row-wise standard deviation (vectorized)
    row_std = df[question_cols].std(axis=1)
    
    # Remove straight-liners (std < 0.5)
    straight_liners = (row_std < 0.5).sum()
    print(f"4. Straight-liners (std < 0.5): {straight_liners}")
    
    df_clean = df[row_std >= 0.5].copy()
    
    # Remove extreme responders (>80% extreme values)
    extreme_mask = (df_clean[question_cols].abs() == 3).sum(axis=1) > len(question_cols) * 0.8
    extreme_count = extreme_mask.sum()
    print(f"5. Extreme responders: {extreme_count}")
    
    df_clean = df_clean[~extreme_mask].copy()
    
    print(f"\n6. Cleaned dataset: {len(df_clean)} rows (removed {len(df) - len(df_clean)})")
    
    return df_clean, question_cols

def reverse_and_score(df, question_cols):
    """Reverse negative items and compute dimension scores."""
    print("\n" + "=" * 60)
    print("STEP 3: Reversing Items & Computing Scores")
    print("=" * 60)
    
    df_scored = df.copy()
    
    # Collect all negative items
    negative_items = []
    for dim, questions in QUESTION_MAPPING.items():
        negative_items.extend(questions['negative'])
    
    # Reverse scores for negative items
    reversed_count = 0
    for col in negative_items:
        if col in df_scored.columns:
            df_scored[col] = -1 * df_scored[col]
            reversed_count += 1
    
    print(f"Reversed {reversed_count} negatively-keyed questions")
    
    # Compute dimension scores
    for dim in ['E_I', 'S_N', 'T_F', 'J_P', 'TURBULENCE']:
        questions = QUESTION_MAPPING[dim]
        all_questions = questions['positive'] + questions['negative']
        existing = [q for q in all_questions if q in df_scored.columns]
        
        if existing:
            df_scored[f'{dim}_score'] = df_scored[existing].mean(axis=1)
    
    # Print dimension statistics
    print("\nDimension Score Statistics:")
    for dim in ['E_I', 'S_N', 'T_F', 'J_P']:
        col = f'{dim}_score'
        print(f"  {dim}: mean={df_scored[col].mean():.3f}, std={df_scored[col].std():.3f}")
    
    return df_scored

def generate_labels(df):
    """Generate MBTI labels from dimension scores."""
    print("\n" + "=" * 60)
    print("STEP 4: Generating MBTI Labels")
    print("=" * 60)
    
    df_labeled = df.copy()
    
    # Compute type based on scores
    def get_type(row):
        e_i = 'E' if row['E_I_score'] > 0 else 'I'
        s_n = 'N' if row['S_N_score'] > 0 else 'S'
        t_f = 'F' if row['T_F_score'] > 0 else 'T'
        j_p = 'J' if row['J_P_score'] > 0 else 'P'
        return e_i + s_n + t_f + j_p
    
    df_labeled['Computed_Type'] = df_labeled.apply(get_type, axis=1)
    
    # Compare with original
    match = (df_labeled['Personality'] == df_labeled['Computed_Type']).sum()
    match_pct = match / len(df_labeled) * 100
    print(f"\nOriginal vs Computed match: {match}/{len(df_labeled)} ({match_pct:.1f}%)")
    
    # Store original and use computed
    df_labeled['Original_Personality'] = df_labeled['Personality']
    df_labeled['Personality'] = df_labeled['Computed_Type']
    
    print("\nComputed Type Distribution:")
    print(df_labeled['Personality'].value_counts())
    
    return df_labeled

def add_features(df):
    """Add additional features for ML."""
    print("\n" + "=" * 60)
    print("STEP 5: Adding Feature Columns")
    print("=" * 60)
    
    df_feat = df.copy()
    
    # Dimension strength (absolute values)
    for dim in ['E_I', 'S_N', 'T_F', 'J_P']:
        df_feat[f'{dim}_strength'] = df_feat[f'{dim}_score'].abs()
    
    # Binary indicators
    df_feat['is_Extraverted'] = (df_feat['E_I_score'] > 0).astype(int)
    df_feat['is_Intuitive'] = (df_feat['S_N_score'] > 0).astype(int)
    df_feat['is_Feeling'] = (df_feat['T_F_score'] > 0).astype(int)
    df_feat['is_Judging'] = (df_feat['J_P_score'] > 0).astype(int)
    
    # Overall consistency
    df_feat['Consistency'] = (
        df_feat['E_I_strength'] + df_feat['S_N_strength'] + 
        df_feat['T_F_strength'] + df_feat['J_P_strength']
    ) / 4
    
    print("Added: strength scores, binary indicators, consistency score")
    
    return df_feat

def balance_dataset(df, target_size=6000):
    """Balance dataset to target size with equal type representation."""
    print("\n" + "=" * 60)
    print("STEP 6: Balancing Dataset")
    print("=" * 60)
    
    samples_per_type = target_size // 16
    
    balanced_dfs = []
    for mbti_type in df['Personality'].unique():
        type_df = df[df['Personality'] == mbti_type]
        
        if len(type_df) >= samples_per_type:
            sampled = type_df.sample(n=samples_per_type, random_state=42)
        else:
            sampled = type_df.sample(n=samples_per_type, replace=True, random_state=42)
        
        balanced_dfs.append(sampled)
    
    df_balanced = pd.concat(balanced_dfs, ignore_index=True)
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"Balanced to {len(df_balanced)} rows ({samples_per_type} per type)")
    print("\nFinal Distribution:")
    print(df_balanced['Personality'].value_counts().sort_index())
    
    return df_balanced

def save_datasets(df, question_cols, output_path):
    """Save cleaned datasets."""
    print("\n" + "=" * 60)
    print("STEP 7: Saving Datasets")
    print("=" * 60)
    
    # Reset Response Id
    df['Response Id'] = range(len(df))
    
    # Full version with all features
    full_cols = ['Response Id'] + question_cols + [
        'E_I_score', 'S_N_score', 'T_F_score', 'J_P_score',
        'E_I_strength', 'S_N_strength', 'T_F_strength', 'J_P_strength',
        'is_Extraverted', 'is_Intuitive', 'is_Feeling', 'is_Judging',
        'Consistency', 'Personality', 'Original_Personality'
    ]
    full_cols = [c for c in full_cols if c in df.columns]
    
    df_full = df[full_cols]
    df_full.to_csv(output_path, index=False, encoding='utf-8')
    print(f"Saved full version: {output_path}")
    
    # Simple version (compatible with original script)
    simple_cols = ['Response Id'] + question_cols + ['Personality']
    df_simple = df[simple_cols]
    simple_path = output_path.replace('.csv', '_simple.csv')
    df_simple.to_csv(simple_path, index=False, encoding='utf-8')
    print(f"Saved simple version: {simple_path}")
    
    return df_full, df_simple

def print_guide():
    """Print interpretation guide."""
    print("\n" + "=" * 60)
    print("MBTI INTERPRETATION GUIDE")
    print("=" * 60)
    print("""
Computing MBTI Type from Scores:
--------------------------------
E_I_score > 0 -> E (Extraversion), else I (Introversion)
S_N_score > 0 -> N (Intuition), else S (Sensing)
T_F_score > 0 -> F (Feeling), else T (Thinking)
J_P_score > 0 -> J (Judging), else P (Perceiving)

Score Interpretation:
---------------------
Near 0: Balanced preference
Near +/-1.5: Moderate preference
Near +/-3: Strong preference

For ML Models:
--------------
- Use question columns as features
- Use 'Personality' as 16-class target
- Or use binary indicators for 4 binary classifications
""")

def main():
    print("\n" + "=" * 60)
    print("MBTI DATASET CLEANING AND ALIGNMENT")
    print("=" * 60)
    
    INPUT_FILE = '16P.csv'
    OUTPUT_FILE = '16P_cleaned.csv'
    TARGET_SIZE = 6000
    
    # Execute pipeline
    df = load_data(INPUT_FILE)
    df, question_cols = analyze_and_clean(df)
    df = reverse_and_score(df, question_cols)
    df = generate_labels(df)
    df = add_features(df)
    df = balance_dataset(df, TARGET_SIZE)
    df_full, df_simple = save_datasets(df, question_cols, OUTPUT_FILE)
    print_guide()
    
    print("\n" + "=" * 60)
    print("CLEANING COMPLETE!")
    print("=" * 60)
    print(f"\nOutput files:")
    print(f"  - {OUTPUT_FILE} (full version)")
    print(f"  - {OUTPUT_FILE.replace('.csv', '_simple.csv')} (simple version)")
    
    return df_full

df_cleaned = main()