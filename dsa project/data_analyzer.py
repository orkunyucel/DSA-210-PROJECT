import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import warnings

# Suppress warnings for clean output
warnings.filterwarnings('ignore')

# Create folder for figures if it doesn't exist
os.makedirs("outputs/figures", exist_ok=True)

def load_and_clean_data(file_path):
    """Load and clean timeout analysis data"""
    
    if not os.path.exists(file_path):
        print(f"Error: Data file {file_path} not found")
        return None
    
    print(f"Loading data from {file_path}...")
    
    # Load data
    df = pd.read_csv(file_path)
    
    # Check for empty DataFrame
    if df.empty:
        print("Data file is empty!")
        return df
    
    print(f"Loaded {len(df)} timeout records")
    
    # Data cleaning
    # Convert boolean columns
    if 'effective' in df.columns and df['effective'].dtype != bool:
        df['effective'] = df['effective'].astype(bool)
    if 'run_terminated' in df.columns and df['run_terminated'].dtype != bool:
        df['run_terminated'] = df['run_terminated'].astype(bool)
    
    # Handle missing values
    for col in df.columns:
        if df[col].dtype in [np.float64, np.int64]:
            missing = df[col].isna().sum()
            if missing > 0:
                print(f"Found {missing} missing values in '{col}' column, filled with zeros")
                df[col] = df[col].fillna(0)
    
    # Check and handle outliers
    for col in ['pre_timeout_oe', 'post_timeout_oe', 'efficiency_change']:
        if col in df.columns:
            # Find outliers (values more than 3 std deviations from the mean)
            mean = df[col].mean()
            std = df[col].std()
            outliers = df[(df[col] > mean + 3*std) | (df[col] < mean - 3*std)]
            
            if len(outliers) > 0:
                print(f"Found {len(outliers)} potential outliers in '{col}' column")
                # Clip outliers
                df[col] = df[col].clip(mean - 3*std, mean + 3*std)
    
    # Create run size categories if not already present
    if 'run_points' in df.columns and 'run_size_bin' not in df.columns:
        df['run_size_bin'] = pd.cut(
            df['run_points'],
            bins=[5, 8, 10, 12, 15, 20, 100],
            labels=['6-7', '8-9', '10-11', '12-14', '15-19', '20+']
        )
    
    # Create score difference bins
    if 'score_diff' in df.columns:
        df['score_situation'] = pd.cut(
            df['score_diff'],
            bins=[-100, -20, -10, -5, 0, 5, 10, 20, 100],
            labels=['Down 20+', 'Down 10-19', 'Down 5-9', 'Down 1-4', 
                   'Up 1-4', 'Up 5-9', 'Up 10-19', 'Up 20+']
        )
    
    return df

def perform_statistical_analysis(results_df):
    """Perform statistical analysis on timeout results"""
    
    if results_df is None or results_df.empty:
        print("No data for analysis")
        return {}
    
    analysis_results = {}
    
    # Overall effectiveness
    total_timeouts = len(results_df)
    effective_timeouts = results_df['effective'].sum()
    effectiveness_rate = effective_timeouts / total_timeouts if total_timeouts > 0 else 0
    
    analysis_results['overall'] = {
        'total_timeouts': total_timeouts,
        'effective_timeouts': effective_timeouts,
        'effectiveness_rate': effectiveness_rate,
        'avg_efficiency_change': results_df['efficiency_change'].mean()
    }
    
    print("\nOverall Results:")
    print(f"Total timeouts analyzed: {total_timeouts}")
    print(f"Effective timeouts: {effective_timeouts} ({effectiveness_rate*100:.1f}%)")
    print(f"Average change in opponent efficiency: {results_df['efficiency_change'].mean():.3f}")
    
    # Determining if the change is statistically significant with t-test
    try:
        t_stat, p_value = stats.ttest_1samp(results_df['efficiency_change'], 0)
        analysis_results['t_test'] = {
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
        
        print("\nStatistical Test Results:")
        print(f"t-statistic: {t_stat:.3f}")
        
        # Use scientific notation for p-value to handle very small values
        if p_value < 1e-12:
            print(f"p-value: {p_value:.6e} (extremely significant)")
        else:
            print(f"p-value: {p_value:.6e}")
            
        print(f"The effect is {'statistically significant' if p_value < 0.05 else 'not statistically significant'}")
    except Exception as e:
        print(f"Error performing t-test: {e}")
        analysis_results['t_test'] = {
            't_statistic': None,
            'p_value': None,
            'significant': False
        }
    
    # Paired t-test: comparing pre-timeout and post-timeout efficiency directly
    try:
        t_paired, p_paired = stats.ttest_rel(results_df['pre_timeout_oe'], results_df['post_timeout_oe'])
        analysis_results['paired_t_test'] = {
            't_statistic': t_paired,
            'p_value': p_paired,
            'significant': p_paired < 0.05
        }
        
        print("\nPaired T-Test (Pre vs Post Timeout Efficiency):")
        print(f"t-statistic: {t_paired:.3f}")
        
        # Use scientific notation for p-value to handle very small values
        if p_paired < 1e-12:
            print(f"p-value: {p_paired:.6e} (extremely significant)")
        else:
            print(f"p-value: {p_paired:.6e}")
            
        print(f"The difference is {'statistically significant' if p_paired < 0.05 else 'not statistically significant'}")
    except Exception as e:
        print(f"Error performing paired t-test: {e}")
        analysis_results['paired_t_test'] = {
            't_statistic': None,
            'p_value': None,
            'significant': False
        }
    
    # Analysis by quarter
    try:
        quarter_analysis = results_df.groupby('quarter').agg({
            'effective': ['count', 'sum', 'mean'],
            'efficiency_change': ['mean'],
            'run_terminated': ['mean']
        })
        analysis_results['by_quarter'] = quarter_analysis
    except Exception as e:
        print(f"Error in quarter analysis: {e}")
        analysis_results['by_quarter'] = pd.DataFrame()
    
    # Analysis by season
    try:
        season_analysis = results_df.groupby('season').agg({
            'effective': ['count', 'sum', 'mean'],
            'efficiency_change': ['mean'],
            'run_terminated': ['mean']
        })
        analysis_results['by_season'] = season_analysis
    except Exception as e:
        print(f"Error in season analysis: {e}")
        analysis_results['by_season'] = pd.DataFrame()
    
    # Analysis by run size
    try:
        if 'run_size_bin' in results_df.columns:
            run_size_analysis = results_df.groupby('run_size_bin').agg({
                'effective': ['count', 'sum', 'mean'],
                'efficiency_change': ['mean'],
                'run_terminated': ['mean']
            })
            analysis_results['by_run_size'] = run_size_analysis
    except Exception as e:
        print(f"Error in run size analysis: {e}")
        analysis_results['by_run_size'] = pd.DataFrame()
    
    return analysis_results

def create_visualizations(results_df, analysis_results):
    """Create visualizations for timeout analysis results"""
    
    if results_df is None or results_df.empty:
        print("No data for visualization")
        return "Failed to create visualizations due to missing data"
    
    # Set Seaborn style
    sns.set(style="whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)
    
    # Function to create and save visualizations
    def create_viz(viz_func, filename, title):
        try:
            viz_func()
            plt.tight_layout()
            plt.savefig(f'outputs/figures/{filename}.png', dpi=300, bbox_inches='tight')
            plt.close()
            return True
        except Exception as e:
            print(f"Error creating {title}: {e}")
            plt.close()
            return False
    
    successful_viz = 0
    total_viz = 0
    
    print("\nCreating visualizations...")
    
    # 1. HISTOGRAM: Efficiency change distribution
    total_viz += 1
    def efficiency_histogram():
        plt.figure(figsize=(12, 8))
        
        # Get efficiency change data and statistics
        eff_change = results_df['efficiency_change']
        mean_change = eff_change.mean()
        median_change = eff_change.median()
        std_change = eff_change.std()
        
        # Create histogram with KDE
        ax = sns.histplot(eff_change, kde=True, bins=25, color='blue')
        
        # Add vertical lines for mean and median
        plt.axvline(mean_change, color='red', linestyle='--', label=f'Mean: {mean_change:.3f}')
        plt.axvline(median_change, color='green', linestyle=':', label=f'Median: {median_change:.3f}')
        
        # Add statistics text box
        stats_text = (
            f"Mean: {mean_change:.3f}\n"
            f"Median: {median_change:.3f}\n"
            f"Std Dev: {std_change:.3f}\n"
            f"Sample Size: {len(eff_change)}"
        )
        plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, 
                fontsize=12, va='top', ha='left',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Add t-test result if available
        if 't_test' in analysis_results and analysis_results['t_test']['t_statistic'] is not None:
            t_stat = analysis_results['t_test']['t_statistic']
            p_val = analysis_results['t_test']['p_value']
            # Format p-value using scientific notation for very small values
            if p_val < 1e-4:
                p_val_str = f"{p_val:.2e}"
            else:
                p_val_str = f"{p_val:.4f}"
                
            t_test_text = f"t-test: t={t_stat:.3f}, p={p_val_str}"
            plt.text(0.05, 0.80, t_test_text, transform=plt.gca().transAxes,
                    fontsize=12, va='top', ha='left',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Add KDE explanation
        plt.text(0.05, 0.70, "Blue curve: Density distribution",
                transform=plt.gca().transAxes, fontsize=12, va='top', ha='left',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.xlabel('Change in Offensive Efficiency (Post-Timeout minus Pre-Timeout)\nPoints Per Possession', fontsize=14)
        plt.ylabel('Frequency', fontsize=14)
        plt.title('Distribution of Opponent Offensive Efficiency Change After Timeouts', fontsize=16)
        plt.legend()
    
    if create_viz(efficiency_histogram, 'efficiency_change_histogram', 
                 'Distribution of Opponent Offensive Efficiency Change After Timeouts'):
        successful_viz += 1
        print("✓ Created efficiency change histogram")
    
    # 2. BOX PLOT: Efficiency change by quarter
    total_viz += 1
    def quarter_boxplot():
        plt.figure(figsize=(12, 8))
        
        # Create boxplot
        ax = sns.boxplot(x='quarter', y='efficiency_change', data=results_df)
        
        # Add sample size and mean information to each quarter
        for i, quarter in enumerate(sorted(results_df['quarter'].unique())):
            quarter_data = results_df[results_df['quarter'] == quarter]
            count = len(quarter_data)
            mean = quarter_data['efficiency_change'].mean()
            
            # Add text above each box
            plt.text(i, plt.ylim()[1]*0.9, f"n = {count}\nMean = {mean:.3f}", 
                     ha='center', va='top', fontsize=11,
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        plt.xlabel('Game Quarter', fontsize=14)
        plt.ylabel('Change in Offensive Efficiency\n(Post-Timeout minus Pre-Timeout)\nPoints Per Possession', fontsize=14)
        plt.title('Offensive Efficiency Change by Quarter', fontsize=16)
    
    if create_viz(quarter_boxplot, 'efficiency_by_quarter_boxplot', 
                 'Offensive Efficiency Change by Quarter'):
        successful_viz += 1
        print("✓ Created quarter boxplot")
    
    # 3. BAR CHART: Timeout effectiveness by team
    total_viz += 1
    def team_effectiveness_bars():
        # Get team effectiveness data
        team_effectiveness = results_df.groupby('opponent_abbr').agg({
            'effective': ['count', 'mean']
        }).reset_index()
        
        # Convert to DataFrame with renamed columns
        team_effectiveness.columns = ['Team', 'Count', 'Effectiveness']
        team_effectiveness = team_effectiveness.sort_values('Effectiveness', ascending=False)
        
        # Filter to teams with at least 10 timeouts
        team_effectiveness = team_effectiveness[team_effectiveness['Count'] >= 10]
        
        if len(team_effectiveness) == 0:
            print("Not enough data to create team effectiveness visualization")
            return None
        
        # Take top 15 teams for better visualization
        team_effectiveness = team_effectiveness.head(15)
        
        plt.figure(figsize=(14, 10))
        
        # Create bar chart
        bars = plt.bar(team_effectiveness['Team'], team_effectiveness['Effectiveness'], 
                      color='skyblue', edgecolor='navy')
        
        # Add data labels on top of each bar
        for i, bar in enumerate(bars):
            count = team_effectiveness.iloc[i]['Count']
            pct = team_effectiveness.iloc[i]['Effectiveness'] * 100
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f"n={count}\n{pct:.1f}%", ha='center', va='bottom', fontsize=10)
        
        # Add horizontal line for overall effectiveness
        overall_effectiveness = results_df['effective'].mean()
        plt.axhline(y=overall_effectiveness, color='red', linestyle='--', 
                   label=f'Overall: {overall_effectiveness:.1%}')
        
        # Format y-axis as percentage
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
        
        plt.xlabel('NBA Team Calling Timeout', fontsize=14)
        plt.ylabel('Timeout Effectiveness Rate\n(% of Timeouts that Reduced Opponent Efficiency)', fontsize=14)
        plt.title('Timeout Effectiveness Rate by NBA Team (Top 15)', fontsize=16)
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, max(team_effectiveness['Effectiveness']) * 1.2)  # Add some space for labels
        plt.legend()
    
    if create_viz(team_effectiveness_bars, 'effectiveness_by_team_bar', 
                 'Timeout Effectiveness Rate by Team'):
        successful_viz += 1
        print("✓ Created team effectiveness bar chart")
    
    # 4. BAR CHART: Timeout effectiveness by season
    total_viz += 1
    def season_effectiveness_bars():
        # Get season effectiveness data
        season_effectiveness = results_df.groupby('season').agg({
            'effective': ['count', 'mean'],
            'efficiency_change': 'mean'
        }).reset_index()
        
        # Convert to DataFrame with renamed columns
        season_effectiveness.columns = ['Season', 'Count', 'Effectiveness', 'Avg_Change']
        
        plt.figure(figsize=(14, 10))
        
        # Create bar chart
        bars = plt.bar(season_effectiveness['Season'], season_effectiveness['Effectiveness'], 
                      color='skyblue', edgecolor='navy')
        
        # Add data labels on top of each bar
        for i, bar in enumerate(bars):
            count = season_effectiveness.iloc[i]['Count']
            pct = season_effectiveness.iloc[i]['Effectiveness'] * 100
            avg_change = season_effectiveness.iloc[i]['Avg_Change']
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f"n={count}\n{pct:.1f}%\nΔ={avg_change:.3f}", ha='center', va='bottom', fontsize=10)
        
        # Add horizontal line for overall effectiveness
        overall_effectiveness = results_df['effective'].mean()
        plt.axhline(y=overall_effectiveness, color='red', linestyle='--', 
                   label=f'Overall: {overall_effectiveness:.1%}')
        
        # Format y-axis as percentage
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
        
        plt.xlabel('NBA Season', fontsize=14)
        plt.ylabel('Timeout Effectiveness Rate\n(% of Timeouts that Reduced Opponent Efficiency)', fontsize=14)
        plt.title('Timeout Effectiveness Rate by NBA Season', fontsize=16)
        plt.ylim(0, max(season_effectiveness['Effectiveness']) * 1.2)  # Add some space for labels
        plt.legend()
    
    if create_viz(season_effectiveness_bars, 'effectiveness_by_season_bar', 
                 'Timeout Effectiveness Rate by Season'):
        successful_viz += 1
        print("✓ Created season effectiveness bar chart")
    
    # 5. SCATTER PLOT: Pre vs post timeout efficiency
    total_viz += 1
    def efficiency_scatter():
        plt.figure(figsize=(12, 10))
        
        # Calculate correlation
        corr = np.corrcoef(results_df['pre_timeout_oe'], results_df['post_timeout_oe'])[0, 1]
        
        # Perform paired t-test
        t_stat, p_value = stats.ttest_rel(results_df['pre_timeout_oe'], results_df['post_timeout_oe'])
        
        # Format p-value using scientific notation for very small values
        if p_value < 1e-4:
            p_val_str = f"{p_value:.2e}"
        else:
            p_val_str = f"{p_value:.4f}"
        
        # Create scatter plot with smaller point size and transparency
        plt.scatter(results_df['pre_timeout_oe'], results_df['post_timeout_oe'], 
                  c=results_df['effective'].map({True: 'green', False: 'red'}),
                  alpha=0.5, s=30, edgecolors='none')
        
        # Add diagonal reference line (y=x)
        min_val = min(results_df['pre_timeout_oe'].min(), results_df['post_timeout_oe'].min())
        max_val = max(results_df['pre_timeout_oe'].max(), results_df['post_timeout_oe'].max())
        lims = [min_val - 0.1, max_val + 0.1]
        plt.plot(lims, lims, 'k--', alpha=0.75, label='No Change Line (y=x)')
        plt.xlim(lims)
        plt.ylim(lims)
        
        # Add statistics text box
        stats_text = (
            f"Sample Size: {len(results_df)}\n"
            f"Correlation: {corr:.3f}\n"
            f"Paired t-test: t={t_stat:.3f}, p={p_val_str}"
        )
        plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, 
                fontsize=12, va='top', ha='left',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, 
                  label='Effective Timeout (Efficiency Decreased)'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, 
                  label='Ineffective Timeout (Efficiency Increased/Unchanged)'),
            Line2D([0], [0], color='k', linestyle='--', label='No Change Line (y=x)')
        ]
        plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0.05, 0.85))
        
        plt.xlabel('Pre-Timeout Offensive Efficiency (Points Per Possession)', fontsize=14)
        plt.ylabel('Post-Timeout Offensive Efficiency (Points Per Possession)', fontsize=14)
        plt.title('Pre-Timeout vs Post-Timeout Offensive Efficiency', fontsize=16)
    
    if create_viz(efficiency_scatter, 'pre_vs_post_timeout_scatter', 
                 'Pre-Timeout vs Post-Timeout Offensive Efficiency'):
        successful_viz += 1
        print("✓ Created pre vs post efficiency scatter plot")
    
    # 6. BAR CHART: Run termination rate by quarter
    total_viz += 1
    def run_termination_by_quarter():
        # Calculate run termination by quarter
        run_termination = results_df.groupby('quarter').agg({
            'run_terminated': ['count', 'mean']
        }).reset_index()
        
        # Convert to DataFrame with renamed columns
        run_termination.columns = ['Quarter', 'Count', 'Termination_Rate']
        
        plt.figure(figsize=(12, 8))
        
        # Create bar chart
        bars = plt.bar(run_termination['Quarter'], run_termination['Termination_Rate'], 
                     color='skyblue', edgecolor='navy')
        
        # Add data labels on top of each bar
        for i, bar in enumerate(bars):
            count = run_termination.iloc[i]['Count']
            pct = run_termination.iloc[i]['Termination_Rate'] * 100
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f"n={count}\n{pct:.1f}%", ha='center', va='bottom', fontsize=10)
        
        # Add horizontal line for overall run termination rate
        overall_rate = results_df['run_terminated'].mean()
        plt.axhline(y=overall_rate, color='red', linestyle='--', 
                  label=f'Overall: {overall_rate:.1%}')
        
        # Format y-axis as percentage
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
        
        plt.xlabel('Game Quarter', fontsize=14)
        plt.ylabel('Run Termination Rate\n(% of Opponent Scoring Runs Stopped After Timeout)', fontsize=14)
        plt.title('Opponent Scoring Run Termination Rate by Quarter', fontsize=16)
        plt.ylim(0, max(run_termination['Termination_Rate']) * 1.2)  # Add some space for labels
        plt.legend()
    
    if create_viz(run_termination_by_quarter, 'run_termination_by_quarter', 
                 'Scoring Run Termination Rate by Quarter'):
        successful_viz += 1
        print("✓ Created run termination by quarter bar chart")
    
    # 7. PIE CHART: Overall timeout effectiveness
    total_viz += 1
    def timeout_effectiveness_pie():
        plt.figure(figsize=(10, 10))
        
        # Calculate effectiveness statistics
        effective_count = int(results_df['effective'].sum())
        ineffective_count = len(results_df) - effective_count
        effective_rate = results_df['effective'].mean()
        ineffective_rate = 1 - effective_rate
        
        # Create data for the pie chart
        labels = [f'Effective\n{effective_count} timeouts ({effective_rate:.1%})', 
                 f'Ineffective\n{ineffective_count} timeouts ({ineffective_rate:.1%})']
        sizes = [effective_rate * 100, ineffective_rate * 100]
        colors = ['green', 'red']
        explode = (0.1, 0)  # Explode the first slice
        
        # Create pie chart
        patches, texts, autotexts = plt.pie(sizes, explode=explode, labels=labels, colors=colors,
                                          autopct='%1.1f%%', shadow=True, startangle=90, 
                                          textprops={'fontsize': 14, 'fontweight': 'bold'})
        
        # Equal aspect ratio ensures the pie chart is circular
        plt.axis('equal')
        
        # Add title
        plt.title('Overall Timeout Effectiveness\n(Ability to Reduce Opponent Offensive Efficiency)', fontsize=16)
    
    if create_viz(timeout_effectiveness_pie, 'timeout_effectiveness_pie', 
                 'Overall Timeout Effectiveness'):
        successful_viz += 1
        print("✓ Created timeout effectiveness pie chart")
    
    # 8. PIE CHART: Run termination
    total_viz += 1
    def run_termination_pie():
        plt.figure(figsize=(10, 10))
        
        # Calculate termination statistics
        terminated_count = int(results_df['run_terminated'].sum())
        continued_count = len(results_df) - terminated_count
        termination_rate = results_df['run_terminated'].mean()
        continuation_rate = 1 - termination_rate
        
        # Create data for the pie chart
        labels = [f'Run Terminated\n{terminated_count} timeouts ({termination_rate:.1%})', 
                 f'Run Continued\n{continued_count} timeouts ({continuation_rate:.1%})']
        sizes = [termination_rate * 100, continuation_rate * 100]
        colors = ['green', 'red']
        explode = (0.1, 0)  # Explode the first slice
        
        # Create pie chart
        patches, texts, autotexts = plt.pie(sizes, explode=explode, labels=labels, colors=colors,
                                         autopct='%1.1f%%', shadow=True, startangle=90,
                                         textprops={'fontsize': 14, 'fontweight': 'bold'})
        
        # Equal aspect ratio ensures the pie chart is circular
        plt.axis('equal')
        
        # Add title
        plt.title('Opponent Scoring Run Termination After Timeout', fontsize=16)
    
    if create_viz(run_termination_pie, 'run_termination_pie', 
                 'Opponent Scoring Run Termination After Timeout'):
        successful_viz += 1
        print("✓ Created run termination pie chart")
    
    # 9. BOX PLOT: Offensive efficiency change by run size
    if 'run_size_bin' in results_df.columns:
        total_viz += 1
        def run_size_boxplot():
            plt.figure(figsize=(12, 8))
            
            # Get unique run sizes and filter out empty ones
            run_sizes = sorted([size for size in results_df['run_size_bin'].unique() if not pd.isna(size)])
            
            # Filter data to only include valid run sizes
            filtered_df = results_df[results_df['run_size_bin'].isin(run_sizes)]
            
            # Create boxplot with filtered data
            ax = sns.boxplot(x='run_size_bin', y='efficiency_change', data=filtered_df, order=run_sizes)
            
            # Add sample size and mean information to each run size
            for i, run_size in enumerate(run_sizes):
                run_data = filtered_df[filtered_df['run_size_bin'] == run_size]
                count = len(run_data)
                mean = run_data['efficiency_change'].mean()
                
                # Add text above each box
                plt.text(i, plt.ylim()[1]*0.9, f"n = {count}\nMean = {mean:.3f}", 
                       ha='center', va='top', fontsize=11,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
            
            plt.xlabel('Opponent Scoring Run Size (Points)', fontsize=14)
            plt.ylabel('Change in Offensive Efficiency\n(Post-Timeout minus Pre-Timeout)\nPoints Per Possession', fontsize=14)
            plt.title('Offensive Efficiency Change by Opponent Scoring Run Size', fontsize=16)
            
            # Adjust figure size to remove excess whitespace
            plt.xlim(-0.5, len(run_sizes)-0.5)
        
        if create_viz(run_size_boxplot, 'efficiency_by_run_size_boxplot', 
                     'Offensive Efficiency Change by Scoring Run Size'):
            successful_viz += 1
            print("✓ Created run size boxplot")
    
    # 10. BAR CHART: Effectiveness by run size
    if 'run_size_bin' in results_df.columns:
        total_viz += 1
        def run_size_effectiveness_bars():
            # Get unique run sizes and filter out empty ones
            run_sizes = sorted([size for size in results_df['run_size_bin'].unique() if not pd.isna(size)])
            
            # Calculate effectiveness by run size
            run_effectiveness_data = []
            
            for run_size in run_sizes:
                run_data = results_df[results_df['run_size_bin'] == run_size]
                count = len(run_data)
                effectiveness = run_data['effective'].mean()
                avg_change = run_data['efficiency_change'].mean()
                
                run_effectiveness_data.append({
                    'Run_Size': run_size,
                    'Count': count,
                    'Effectiveness': effectiveness,
                    'Avg_Change': avg_change
                })
            
            run_effectiveness = pd.DataFrame(run_effectiveness_data)
            
            # Only proceed if we have data
            if len(run_effectiveness) == 0:
                print("No run size data available")
                return None
            
            plt.figure(figsize=(12, 8))
            
            # Create bar chart
            bars = plt.bar(run_effectiveness['Run_Size'], run_effectiveness['Effectiveness'], 
                         color='skyblue', edgecolor='navy')
            
            # Add data labels on top of each bar
            for i, bar in enumerate(bars):
                count = run_effectiveness.iloc[i]['Count']
                pct = run_effectiveness.iloc[i]['Effectiveness'] * 100
                avg_change = run_effectiveness.iloc[i]['Avg_Change']
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                       f"n={count}\n{pct:.1f}%\nΔ={avg_change:.3f}", ha='center', va='bottom', fontsize=10)
            
            # Add horizontal line for overall effectiveness
            overall_effectiveness = results_df['effective'].mean()
            plt.axhline(y=overall_effectiveness, color='red', linestyle='--', 
                      label=f'Overall: {overall_effectiveness:.1%}')
            
            # Format y-axis as percentage
            plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
            
            plt.xlabel('Opponent Scoring Run Size (Points)', fontsize=14)
            plt.ylabel('Timeout Effectiveness Rate\n(% of Timeouts that Reduced Opponent Efficiency)', fontsize=14)
            plt.title('Timeout Effectiveness Rate by Opponent Scoring Run Size', fontsize=16)
            plt.ylim(0, max(run_effectiveness['Effectiveness']) * 1.2)  # Add some space for labels
            plt.legend()
            
            # Adjust figure size to remove excess whitespace
            plt.xlim(-0.5, len(run_sizes)-0.5)
        
        if create_viz(run_size_effectiveness_bars, 'effectiveness_by_run_size_bar', 
                     'Timeout Effectiveness Rate by Scoring Run Size'):
            successful_viz += 1
            print("✓ Created run size effectiveness bar chart")
    
    # 11. COMBINED VISUALIZATION: Pre vs Post timeout metrics comparison
    total_viz += 1
    def pre_post_metrics_comparison():
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # 1. Field Goal Percentage Comparison
        pre_fg = results_df['pre_timeout_fg_pct']
        post_fg = results_df['post_timeout_fg_pct']
        
        # Calculate statistics
        pre_fg_mean = pre_fg.mean()
        post_fg_mean = post_fg.mean()
        t_stat_fg, p_val_fg = stats.ttest_rel(pre_fg, post_fg)
        
        # Format p-value using scientific notation for very small values
        if p_val_fg < 1e-4:
            p_val_fg_str = f"{p_val_fg:.2e}"
        else:
            p_val_fg_str = f"{p_val_fg:.4f}"
        
        # Create bar chart
        x = ['Pre-Timeout', 'Post-Timeout']
        y = [pre_fg_mean, post_fg_mean]
        
        ax1.bar(x, y, color=['blue', 'red'], alpha=0.7)
        
        # Add data labels
        ax1.text(0, pre_fg_mean+0.01, f"{pre_fg_mean:.1%}", ha='center', fontsize=12)
        ax1.text(1, post_fg_mean+0.01, f"{post_fg_mean:.1%}", ha='center', fontsize=12)
        
        # Add statistical test results
        if p_val_fg < 0.05:
            significance = "Significant difference"
        else:
            significance = "Not significant"
            
        ax1.text(0.5, max(y)*1.1, f"p={p_val_fg_str}\n{significance}", 
               ha='center', fontsize=12,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Format y-axis as percentage
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
        
        ax1.set_ylim(0, max(y)*1.2)  # Add some space for labels
        ax1.set_title('Field Goal Percentage Comparison', fontsize=14)
        ax1.set_ylabel('Field Goal Percentage', fontsize=12)
        
        # 2. True Shooting Percentage Comparison
        pre_ts = results_df['pre_timeout_ts']
        post_ts = results_df['post_timeout_ts']
        
        # Calculate statistics
        pre_ts_mean = pre_ts.mean()
        post_ts_mean = post_ts.mean()
        t_stat_ts, p_val_ts = stats.ttest_rel(pre_ts, post_ts)
        
        # Format p-value using scientific notation for very small values
        if p_val_ts < 1e-4:
            p_val_ts_str = f"{p_val_ts:.2e}"
        else:
            p_val_ts_str = f"{p_val_ts:.4f}"
        
        # Create bar chart
        y = [pre_ts_mean, post_ts_mean]
        
        ax2.bar(x, y, color=['blue', 'red'], alpha=0.7)
        
        # Add data labels
        ax2.text(0, pre_ts_mean+0.01, f"{pre_ts_mean:.1%}", ha='center', fontsize=12)
        ax2.text(1, post_ts_mean+0.01, f"{post_ts_mean:.1%}", ha='center', fontsize=12)
        
        # Add statistical test results
        if p_val_ts < 0.05:
            significance = "Significant difference"
        else:
            significance = "Not significant"
            
        ax2.text(0.5, max(y)*1.1, f"p={p_val_ts_str}\n{significance}", 
               ha='center', fontsize=12,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Format y-axis as percentage
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
        
        ax2.set_ylim(0, max(y)*1.2)  # Add some space for labels
        ax2.set_title('True Shooting Percentage Comparison', fontsize=14)
        ax2.set_ylabel('True Shooting Percentage', fontsize=12)
        
        plt.suptitle('Opponent Shooting Efficiency: Pre-Timeout vs Post-Timeout', fontsize=16)
    
    if create_viz(pre_post_metrics_comparison, 'pre_post_shooting_comparison', 
                 'Opponent Shooting Efficiency Comparison'):
        successful_viz += 1
        print("✓ Created pre vs post shooting metrics comparison")
    
    # 12. STACKED BAR CHART: Run termination by season
    total_viz += 1
    def run_termination_by_season():
        # Calculate run termination by season
        season_data = results_df.groupby('season').agg({
            'run_terminated': ['count', 'sum']
        }).reset_index()
        
        # Convert to DataFrame with renamed columns
        season_data.columns = ['Season', 'Total', 'Terminated']
        season_data['Continued'] = season_data['Total'] - season_data['Terminated']
        
        plt.figure(figsize=(14, 8))
        
        # Create stacked bar chart
        plt.bar(season_data['Season'], season_data['Terminated'], 
              color='green', label='Run Terminated')
        plt.bar(season_data['Season'], season_data['Continued'], 
              bottom=season_data['Terminated'], color='red', label='Run Continued')
        
        # Add data labels
        for i, season in enumerate(season_data['Season']):
            total = season_data.iloc[i]['Total']
            terminated = season_data.iloc[i]['Terminated']
            term_rate = terminated / total
            
            # Add percentage on green bars (clearer to know it belongs to green)
            plt.text(i, terminated/2, f"{term_rate:.1%}", 
                   ha='center', va='center', color='white', fontweight='bold',
                   fontsize=12)
            
            # Add total count on top
            plt.text(i, total + 1, f"n={total}", 
                   ha='center', va='bottom')
        
        plt.xlabel('NBA Season', fontsize=14)
        plt.ylabel('Number of Timeouts', fontsize=14)
        plt.title('Opponent Scoring Run Termination by Season', fontsize=16)
        
        # Add a legend with clear labels
        plt.legend(loc='upper left', title="Percentage shows Run Termination Rate")
    
    if create_viz(run_termination_by_season, 'run_termination_by_season', 
                 'Scoring Run Termination by Season'):
        successful_viz += 1
        print("✓ Created run termination by season stacked bar chart")
    
    print(f"\nSuccessfully created {successful_viz} out of {total_viz} visualizations")
    return f"Successfully created {successful_viz}/{total_viz} visualizations"

def generate_summary_report(results_df, analysis_results):
    """Generate a summary report of the analysis"""
    
    if results_df is None or results_df.empty or not analysis_results:
        return "No analysis results to report."
    
    # Overall results
    overall = analysis_results.get('overall', {})
    total_timeouts = overall.get('total_timeouts', 0)
    effective_timeouts = overall.get('effective_timeouts', 0)
    effectiveness_rate = overall.get('effectiveness_rate', 0)
    avg_efficiency_change = overall.get('avg_efficiency_change', 0)
    
    # T-test results
    t_test = analysis_results.get('t_test', {})
    t_statistic = t_test.get('t_statistic', None)
    p_value = t_test.get('p_value', None)
    significant = t_test.get('significant', False)
    
    # Create summary report
    report = [
        "# NBA Timeout Analysis Summary Report",
        "",
        "## Overall Results",
        "",
        f"* Total timeouts analyzed: {total_timeouts}",
        f"* Effective timeouts: {effective_timeouts} ({effectiveness_rate*100:.1f}%)",
        f"* Average change in opponent offensive efficiency: {avg_efficiency_change:.3f} points per possession",
        "",
        "## Statistical Significance",
        ""
    ]
    
    if t_statistic is not None and p_value is not None:
        # Format p-value using scientific notation for very small values
        if p_value < 1e-4:
            p_value_str = f"{p_value:.6e}"
        else:
            p_value_str = f"{p_value:.6f}"
            
        report.extend([
            f"* t-statistic: {t_statistic:.3f}",
            f"* p-value: {p_value_str}",
            f"* Result: The effect of timeouts on opponent offensive efficiency is {'statistically significant' if significant else 'not statistically significant'}",
            ""
        ])
    else:
        report.extend([
            "* Statistical test could not be performed due to insufficient data",
            ""
        ])
    
    # Hypothesis test
    report.extend([
        "## Hypothesis Test Results",
        "",
        "**Null Hypothesis (H₀):** When the opponent team makes a scoring run and a timeout is called, the opponent's average offensive efficiency from the start of the period to the timeout equals its average offensive efficiency from the timeout to the end of the period.",
        ""
    ])
    
    if t_statistic is not None:
        if significant:
            if avg_efficiency_change < 0:
                report.append("**Result:** p-value < 0.05, rejecting the null hypothesis. Timeouts have been found to significantly reduce opponent offensive efficiency.")
            else:
                report.append("**Result:** p-value < 0.05, rejecting the null hypothesis. Timeouts have been found to significantly affect opponent offensive efficiency, but surprisingly, efficiency increases after timeouts rather than decreases.")
        else:
            report.append("**Result:** p-value > 0.05, failing to reject the null hypothesis. There is insufficient evidence to conclude that timeouts significantly affect opponent offensive efficiency.")
    
    # Conclusion
    report.extend([
        "",
        "## Conclusion",
        ""
    ])
    
    if t_statistic is not None and significant:
        if avg_efficiency_change < 0:
            report.append("The analysis demonstrates that timeouts are effective in disrupting opponent momentum. After an opponent's scoring run, timeouts lead to a statistically significant decrease in their offensive efficiency. This provides empirical evidence supporting the common basketball coaching practice of calling timeouts to \"stop the bleeding\" when the opponent is on a run.")
        else:
            report.append("Contrary to conventional basketball wisdom, the analysis reveals that timeouts might actually enhance opponent momentum. Following an opponent's scoring run, timeouts lead to a statistically significant increase in their offensive efficiency. This unexpected finding challenges the traditional coaching strategy of calling timeouts to disrupt opponent momentum.")
    else:
        report.append("The analysis does not provide statistical evidence that timeouts significantly disrupt opponent momentum. While there is a slight change in offensive efficiency after timeouts, this difference is not statistically significant. This suggests that the common practice of calling timeouts to \"stop the bleeding\" may be less effective than traditionally believed.")
    
    # Limitations and future work
    report.extend([
        "",
        "## Limitations and Future Work",
        "",
        "This analysis has several limitations that should be considered:",
        "",
        "1. **Sample Size**: The analysis is based on a subset of NBA games and may not perfectly represent all timeout situations.",
        "2. **Causality**: While we observe changes in efficiency after timeouts, these changes could be influenced by factors beyond the timeout itself.",
        "3. **Context**: Not all timeouts are called solely to disrupt momentum; coaches may call timeouts for strategic purposes or to rest players.",
        "4. **Data Quality**: Play-by-play data, especially from older seasons, may contain inconsistencies or missing information.",
        "",
        "Future research could explore:",
        "",
        "- Comparing timeout effectiveness in playoff games versus regular season games",
        "- Examining the impact of different coaches on timeout effectiveness",
        "- Analyzing different types of timeouts (full vs. 20-second)",
        "- Investigating specific plays drawn up during timeouts",
        "- Expanding the analysis to include international leagues and college basketball"
    ])
    
    # Save the report to a markdown file
    with open('outputs/timeout_analysis_report.md', 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    print("\nSummary report generated and saved to 'outputs/timeout_analysis_report.md'")
    return "Report generated successfully"

# Main function
def run_analysis(data_file='outputs/timeout_analysis_results.csv'):
    """Run the full timeout analysis pipeline"""
    
    # Load and clean data
    results_df = load_and_clean_data(data_file)
    
    if results_df is None or results_df.empty:
        print("Error: No valid data for analysis. Please run data collection first.")
        return None, None
    
    # Perform statistical analysis
    print("\nPerforming statistical analysis...")
    analysis_results = perform_statistical_analysis(results_df)
    
    # Create visualizations
    print("\nCreating visualizations...")
    viz_result = create_visualizations(results_df, analysis_results)
    
    # Generate summary report
    print("\nGenerating summary report...")
    report_result = generate_summary_report(results_df, analysis_results)
    
    print("\nAnalysis complete! Results saved to 'outputs' directory.")
    return results_df, analysis_results

# Try to load both final and partial result files
def try_analysis():
    """Try to analyze data from either the final or partial results file"""
    
    # First try the final results file
    if os.path.exists('outputs/timeout_analysis_results.csv'):
        print("Using final results file")
        return run_analysis('outputs/timeout_analysis_results.csv')
    
    # If not available, try the partial results file
    elif os.path.exists('outputs/timeout_analysis_results_partial.csv'):
        print("Using partial results file")
        return run_analysis('outputs/timeout_analysis_results_partial.csv')
    
    # If neither is available
    else:
        print("No results file found. Please run data collection first.")
        return None, None

# Run the analysis when this script is executed
if __name__ == "__main__":
    try_analysis()