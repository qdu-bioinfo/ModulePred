import csv


def process_gene_network(input_csv, output_dict_txt, output_reverse_dict_txt, output_edges_csv):

    gene_pairs = []
    unique_genes = set()

    with open(input_csv, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if len(row) >= 2:
                gene1, gene2 = row[0].strip(), row[1].strip()
                if gene1 and gene2:
                    gene_pairs.append((gene1, gene2))
                    unique_genes.update([gene1, gene2])


    sorted_genes = sorted(unique_genes)
    gene_to_id = {gene: idx for idx, gene in enumerate(sorted_genes)}
    id_to_gene = {idx: gene for gene, idx in gene_to_id.items()}


    with open(output_dict_txt, 'w') as f:
        for gene, idx in gene_to_id.items():
            f.write(f"{gene} {idx}\n")  # 空格分隔


    with open(output_reverse_dict_txt, 'w') as f:
        for idx, gene in id_to_gene.items():
            f.write(f"{idx} {gene}\n")  # 空格分隔


    with open(output_edges_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["gene1", "gene2"])  # 表头
        for gene1, gene2 in gene_pairs:
            writer.writerow([gene_to_id[gene1], gene_to_id[gene2]])

    return gene_to_id, id_to_gene, gene_pairs



input_csv = "../data/gene_gene.csv" #
output_dict_txt = "./Output/gene_mapping.txt"
output_edges_csv = "./Output/gene_gene_num.csv"
output_reverse_dict_txt = "./Output/gene_num_return_dict.txt"

gene_to_id, id_to_gene, edges = process_gene_network(
    input_csv, output_dict_txt, output_reverse_dict_txt, output_edges_csv
)

print(f"处理完成！共 {len(gene_to_id)} 个唯一基因，{len(edges)} 条连接关系")

