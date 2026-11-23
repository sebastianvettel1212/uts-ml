import json
import io
import sys
import base64
import nbformat as nbf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def run_notebook_inprocess(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        nb = nbf.read(f, as_version=4)

    ns = {
        '__name__': '__main__',
        'display': lambda x: print(x.to_string() if hasattr(x, 'to_string') else str(x))
    }
    
    exec_count = 1
    total_code_cells = sum(1 for c in nb.cells if c.cell_type == 'code')
    print(f"Memulai eksekusi {total_code_cells} sel kode secara in-process...")

    for i, cell in enumerate(nb.cells):
        if cell.cell_type != 'code':
            continue

        source = cell.source
        # Bersihkan magic commands IPython
        cleaned_lines = []
        for line in source.split('\n'):
            if line.strip().startswith('%'):
                cleaned_lines.append('# ' + line)
            else:
                cleaned_lines.append(line)
        clean_code = '\n'.join(cleaned_lines)

        stdout_trap = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = stdout_trap

        cell_outputs = []
        # Tutup figur lama sebelum eksekusi sel
        plt.close('all')

        try:
            exec(clean_code, ns)
        except Exception as e:
            sys.stdout = old_stdout
            print(f"Error di sel {i}: {e}")
            raise e
        finally:
            sys.stdout = old_stdout

        out_text = stdout_trap.getvalue()
        if out_text:
            cell_outputs.append(nbf.v4.new_output(
                output_type='stream',
                name='stdout',
                text=out_text
            ))

        # Cek apakah ada figur matplotlib yang dibuat
        fignums = plt.get_fignums()
        for fignum in fignums:
            fig = plt.figure(fignum)
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
            buf.seek(0)
            img_b64 = base64.b64encode(buf.read()).decode('utf-8')
            plt.close(fig)
            cell_outputs.append(nbf.v4.new_output(
                output_type='display_data',
                data={
                    'image/png': img_b64,
                    'text/plain': '<Figure size ...>'
                },
                metadata={}
            ))

        cell.outputs = cell_outputs
        cell.execution_count = exec_count
        print(f"  [Sel {exec_count}/{total_code_cells}] Berhasil dieksekusi ({len(cell_outputs)} outputs)")
        exec_count += 1

    with open(output_path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print(f"Selesai! Notebook tersimpan lengkap dengan seluruh output di '{output_path}'.")

if __name__ == '__main__':
    run_notebook_inprocess('task_3_clustering_tahap_0_tahap_c.ipynb', 'task_3_clustering_tahap_0_tahap_c.ipynb')
