import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import argparse
import numpy as np

def load_and_preprocess_data(log_dir):
    """
    Loads all CSV logs from a directory, combines them, and adds
    'topology' and 'algorithm' columns based on filenames.
    """
    all_files = glob.glob(os.path.join(log_dir, '*.csv'))
    if not all_files:
        print(f"Error: No CSV files found in directory '{log_dir}'.")
        return pd.DataFrame()

    df_list = []
    for file in all_files:
        try:
            df = pd.read_csv(file)
            filename = os.path.basename(file).replace('.csv', '')
            parts = filename.split('_')
            df['topology'] = parts[0]
            # Handle cases like 'small_world'
            if len(parts) > 2:
                df['topology'] = '_'.join(parts[:-1])
                df['algorithm'] = parts[-1]
            else:
                df['algorithm'] = parts[1]
            df_list.append(df)
        except Exception as e:
            print(f"Warning: Could not process file {file}. Error: {e}")
    
    if not df_list:
        return pd.DataFrame()
        
    return pd.concat(df_list, ignore_index=True)

def plot_comparative_learning_curves(df, output_dir):
    """
    Plots the rolling average of success rate and reward for each topology,
    comparing all algorithms on a single set of axes.
    """
    if df.empty: return
    
    print("Generating comparative learning curve plots...")
    
    window = 100
    # Calculate rolling average for success (all algorithms have this)
    df['success_rate_smooth'] = df.groupby(['topology', 'algorithm'])['success'].transform(lambda s: s.rolling(window).mean())

    # --- SUCCESS RATE PLOT ---
    g = sns.FacetGrid(df, col="topology", hue="algorithm", col_wrap=3, height=5, aspect=1.5, legend_out=True)
    g.map(sns.lineplot, "episode", "success_rate_smooth", alpha=0.8).add_legend()
    g.set_titles("Topology: {col_name}")
    g.set_axis_labels("Episode", f"Success Rate (Rolling Avg, Window={window})")
    plt.tight_layout()
    output_path = os.path.join(output_dir, "comparative_success_rate.png")
    plt.savefig(output_path, dpi=300)
    print(f"Saved success rate plot to {output_path}")
    plt.close()

    # --- REWARD PLOT (MODIFIED TO BE ROBUST) ---
    # Only plot rewards for algorithms that have a 'reward' column (e.g., DRL)
    if 'reward' in df.columns:
        reward_df = df.dropna(subset=['reward']) # Filter out rows where reward is not applicable
        if not reward_df.empty:
            reward_df['reward_smooth'] = reward_df.groupby(['topology', 'algorithm'])['reward'].transform(lambda s: s.rolling(window).mean())

            g_reward = sns.FacetGrid(reward_df, col="topology", hue="algorithm", col_wrap=3, height=5, aspect=1.5, legend_out=True)
            g_reward.map(sns.lineplot, "episode", "reward_smooth", alpha=0.8).add_legend()
            g_reward.set_titles("Topology: {col_name}")
            g_reward.set_axis_labels("Episode", f"Reward (Rolling Avg, Window={window})")
            plt.tight_layout()
            output_path = os.path.join(output_dir, "comparative_reward.png")
            plt.savefig(output_path, dpi=300)
            print(f"Saved reward plot to {output_path}")
            plt.close()
        else:
            print("No valid reward data found to plot.")
    else:
        print("'reward' column not found in any logs, skipping reward plot.")


def plot_final_performance_distribution(df, output_dir, last_n=500):
    """
    Creates box plots to compare the distribution of key metrics over the
    last N episodes of the simulation.
    """
    if df.empty: return
    
    print(f"Generating final performance distribution plots (last {last_n} episodes)...")
    
    # Filter for the final N episodes
    max_episode = df['episode'].max()
    final_df = df[df['episode'] >= max_episode - last_n]
    
    # Plot Fidelity Distribution
    plt.figure(figsize=(15, 8))
    sns.boxplot(data=final_df[final_df['success']==1], x='topology', y='fidelities', hue='algorithm')
    plt.title(f'Distribution of Fidelity on Successful Routes (Last {last_n} Episodes)')
    plt.ylabel('Fidelity')
    plt.xlabel('Topology')
    plt.ylim(0.5, 1.0)
    plt.tight_layout()
    output_path = os.path.join(output_dir, "final_fidelity_distribution.png")
    plt.savefig(output_path, dpi=300)
    print(f"Saved fidelity distribution plot to {output_path}")
    plt.close()

    # Plot Path Length Distribution
    plt.figure(figsize=(15, 8))
    sns.boxplot(data=final_df[final_df['success']==1], x='topology', y='path_lengths', hue='algorithm')
    plt.title(f'Distribution of Path Length on Successful Routes (Last {last_n} Episodes)')
    plt.ylabel('Path Length (Hops)')
    plt.xlabel('Topology')
    plt.tight_layout()
    output_path = os.path.join(output_dir, "final_path_length_distribution.png")
    plt.savefig(output_path, dpi=300)
    print(f"Saved path length distribution plot to {output_path}")
    plt.close()

def generate_summary_report(df, output_dir, last_n=500):
    """
    Calculates final performance metrics and saves them as a CSV and Markdown table.
    """
    if df.empty: return
    
    print("Generating summary report...")
    
    # Filter for final performance
    max_episode = df['episode'].max()
    final_df = df[df['episode'] >= max_episode - last_n]
    
    # Group and aggregate
    summary = final_df.groupby(['topology', 'algorithm']).agg(
        final_success_rate=('success', 'mean'),
        avg_fidelity_on_success=('fidelities', lambda x: x[final_df.loc[x.index, 'success'] == 1].mean()),
        std_fidelity_on_success=('fidelities', lambda x: x[final_df.loc[x.index, 'success'] == 1].std()),
        avg_path_length_on_success=('path_lengths', lambda x: x[final_df.loc[x.index, 'success'] == 1].mean()),
    ).reset_index()

    # Format for readability
    summary['final_success_rate'] = (summary['final_success_rate'] * 100).round(2)
    summary = summary.round(3)
    
    # Save as CSV
    csv_path = os.path.join(output_dir, 'summary_report.csv')
    summary.to_csv(csv_path, index=False)
    print(f"Summary CSV saved to {csv_path}")

    # Save as Markdown
    md_path = os.path.join(output_dir, 'summary_report.md')
    summary.to_markdown(md_path, index=False)
    print(f"Summary Markdown saved to {md_path}")

def main():
    parser = argparse.ArgumentParser(description="Analyze results from quantum routing simulations.")
    parser.add_argument("log_dir", type=str, nargs='?', default="logs", help="Directory containing the CSV log files. Defaults to 'logs'.")
    parser.add_argument("--output_dir", type=str, default="analysis_results", help="Directory to save analysis plots and reports.")
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    sns.set_theme(style="whitegrid", palette="viridis")

    # 1. Load and combine data
    full_df = load_and_preprocess_data(args.log_dir)
    
    if full_df.empty:
        print("No data loaded. Exiting analysis.")
        return

    # 2. Generate plots
    plot_comparative_learning_curves(full_df, args.output_dir)
    plot_final_performance_distribution(full_df, args.output_dir)

    # 3. Generate summary report
    generate_summary_report(full_df, args.output_dir)

    print("\nAnalysis complete.")
    print(f"All reports and plots are saved in the '{args.output_dir}' directory.")

if __name__ == '__main__':
    main()

