import os
import csv

def main():
    # Define directories
    base_dir = '/home/mr/Documents/School/Thesis/results'
    # Output file will be placed in the Thesis folder
    output_file = '/home/mr/Documents/School/Thesis/results/results_sum.md'
    
    results = []
    all_metrics = set()
    
    # Find all results.csv files
    for root, dirs, files in os.walk(base_dir):
        if 'results.csv' in files:
            csv_path = os.path.join(root, 'results.csv')
            exp_name = os.path.basename(root)
            
            with open(csv_path, 'r', newline='', encoding='utf-8') as f:
                reader = csv.reader(f)
                header = next(reader, None)
                if not header:
                    continue
                
                # Identify indexes of metrics (exclude 'run' and 'seed')
                metric_cols = []
                for i, col in enumerate(header):
                    if col not in ('run', 'seed'):
                        metric_cols.append((i, col))
                        all_metrics.add(col)
                
                # Find mean row and count repetitions
                mean_row = None
                repetitions = 0
                run_idx = header.index('run') if 'run' in header else -1
                
                for row in reader:
                    if run_idx != -1 and len(row) > run_idx:
                        if row[run_idx].startswith('run_'):
                            repetitions += 1
                        elif row[run_idx] == 'mean':
                            mean_row = row
                
                if mean_row:
                    exp_data = {
                        'experiment_name': exp_name,
                        'repetitions': repetitions
                    }
                    for idx, col in metric_cols:
                        if idx < len(mean_row):
                            exp_data[col] = mean_row[idx]
                    results.append(exp_data)
    
    # Sort metrics to have a consistent order (we can try to preserve original order if possible, 
    # but sorting is safer across multiple files with potentially different columns)
    # Let's define a custom sort order for common metrics
    def sort_key(x):
        # Optional: prioritize certain known columns if they exist
        priority = {'UAR': 0, 'accuracy': 1, 'macro_f1': 2, 'weighted_f1': 3}
        return (priority.get(x, 100), x)

    metric_list = sorted(list(all_metrics), key=sort_key)
    final_headers = ['experiment_name', 'repetitions'] + metric_list
    
    # Write to central text file
    max_name_len = max((len(r['experiment_name']) for r in results), default=15)
    max_name_len = max(max_name_len, len('experiment_name'))
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('```yaml enhanced-tables\n')
        f.write('columns:\n')
        f.write('  repetitions:\n    type: number\n')
        for m in metric_list:
            f.write(f'  {m}:\n    type: number\n')
        f.write('```\n\n')
        
        header_str = f"| {'experiment_name'.ljust(max_name_len)} | {'repetitions'.rjust(11)}"
        for m in metric_list:
            col_width = max(10, len(m))
            header_str += f" | {m.rjust(col_width)}"
        header_str += " |"
        f.write(header_str + '\n')
        
        sep_str = f"| {'-' * max_name_len} | {'-' * 11}"
        for m in metric_list:
            col_width = max(10, len(m))
            sep_str += f" | {'-' * col_width}"
        sep_str += " |"
        f.write(sep_str + '\n')
        
        # Sort results by experiment name for better readability
        results.sort(key=lambda x: x['experiment_name'])
        
        for row in results:
            row_str = f"| {row['experiment_name'].ljust(max_name_len)} | {str(row.get('repetitions', '')).rjust(11)}"
            for m in metric_list:
                val = str(row.get(m, ''))
                try:
                    val = f"{float(val):.4f}"
                except ValueError:
                    pass
                col_width = max(10, len(m))
                row_str += f" | {val.rjust(col_width)}"
            row_str += " |"
            f.write(row_str + '\n')

    print(f"Successfully processed {len(results)} experiments.")
    print(f"Summary saved to {output_file}")

if __name__ == '__main__':
    main()
