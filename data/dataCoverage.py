import json

def main():
    train_data_path = './data/dev.json'
    
    homonym_count = 1
    homonym_instances = {}
    with open(train_data_path, 'r', encoding='utf-8') as f:
        file_dict = json.load(f)
        homonym = file_dict["0"]['homonym']
        count = 0
        for item in file_dict.values():
            if homonym != item['homonym']:
                homonym_instances[homonym] = count
                homonym_count += 1
                homonym = item['homonym']
                count = 0
            count += 1
    
    print(homonym_count)
    print(sum(homonym_instances.values()))


if __name__ == '__main__':
    main()
