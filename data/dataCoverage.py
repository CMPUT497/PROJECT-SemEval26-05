import json

def main():
    train_data_path = ['./data/train.json', './data/dev.json']
    
    for path in train_data_path:
        homonym_count = 1
        homonym_pair_count = 1
        homonym_instances = {}
        homonym_meaning_pairs = {}
        print("Coverage of Data in file: ", path)
        
        with open(path, 'r', encoding='utf-8') as f:
            file_dict = json.load(f)
            homonym = file_dict["0"]['homonym']
            count = 0
            meaning = file_dict["0"]['judged_meaning']
            for item in file_dict.values():
                if homonym != item['homonym']:
                    homonym_instances[homonym] = count
                    homonym_count += 1
                    homonym = item['homonym']
                    count = 0
                count += 1
                if (item['homonym'], item['judged_meaning']) not in homonym_meaning_pairs:
                    homonym_meaning_pairs[(item['homonym'], item['judged_meaning'])] = 0
                    homonym_pair_count += 1
                else:
                    homonym_meaning_pairs[(item['homonym'], item['judged_meaning'])] += 1
                                
        print("Total number of homonyms:", homonym_count)
        print("Total number of instances:", sum(homonym_instances.values()))
               
        # Find homonyms with lowest and highest counts
        lowest_instances_for_homonym = float('inf')
        lowest_count_homonym = ""
        highest_instances_for_homonym = 0
        highest_count_homonym = ""
        for homonym, instances in homonym_instances.items():
            if instances < lowest_instances_for_homonym:
                lowest_instances_for_homonym = instances
                lowest_count_homonym = homonym
            if instances > highest_instances_for_homonym:
                highest_instances_for_homonym = instances
                highest_count_homonym = homonym

        print(f"Homonym with lowest count: {lowest_count_homonym} -> {lowest_instances_for_homonym}")
        print(f"Homonym with highest count: {highest_count_homonym} -> {highest_instances_for_homonym}")
        print()

        # Find pairs with lowest and highest counts
        lowest_instances_for_pair = float('inf')
        lowest_count_pair = ()
        highest_instances_for_pair = 0
        highest_count_pair = ()
        for pair, instances in homonym_meaning_pairs.items():
            if instances < lowest_instances_for_pair:
                lowest_instances_for_pair = instances
                lowest_count_pair = pair
            if instances > highest_instances_for_pair:
                highest_instances_for_pair = instances
                highest_count_pair = pair

        print("Total number of homonym-meaning pairs:", homonym_pair_count)
        print("Total number of instances across all pairs:", sum(homonym_meaning_pairs.values()))

        print(f"Homonym-meaning pair with lowest count: {lowest_count_pair} -> {lowest_instances_for_pair}")
        print(f"Homonym-meaning pair with highest count: {highest_count_pair} -> {highest_instances_for_pair}")
        print()
        print()
        

if __name__ == '__main__':
    main()
