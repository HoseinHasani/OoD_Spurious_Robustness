sub_dataloader_dict = {}
            DATA_PATH = "/home/hasani2/R/Datasets/Waterbirds_dataset/waterbird_complete"+str(correlation)+"_forest2water2"
            wb_train_loader, wb_val_loader, wb_test_loader = get_waterbird_loaders(path=DATA_PATH,
                                                            batch_size=batch_size)

            sub_dataloader_dict['train'] = wb_train_loader
            sub_dataloader_dict['val'] = wb_val_loader
            sub_dataloader_dict['test'] = wb_test_loader

            dataloader_dict['id'] = sub_dataloader_dict
            dataloader_dict['ood'] = {}
            
            scale = 256.0/224.0
            target_resolution = (224, 224)
            large_transform = transforms.Compose([
                transforms.Resize((int(target_resolution[0]*scale), int(target_resolution[1]*scale))),
                transforms.CenterCrop(target_resolution),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

            testsetout = ImageFolder("/home/hasani2/R/Datasets/OOD_Datasets/placesbg/",transform=large_transform)
            np.random.seed(seed)
            subset_indices = np.random.choice(len(testsetout), 2000, replace=False)
            subset_val_indices = subset_indices[:int(0.2 * len(subset_indices))]  
            subset_near_indices = subset_indices[int(0.2 * len(subset_indices)):]  
            subset_val = torch.utils.data.Subset(testsetout, subset_val_indices)
            # sub_dataloader_dict_val = {'sp_waterbirds': torch.utils.data.DataLoader(subset_val, batch_size=64, shuffle=False, num_workers=4)}
            # dataloader_dict['ood']['val']= sub_dataloader_dict_val

            testsetout2 = ImageFolder("/home/hasani2/R/Datasets/OOD_Datasets/stable_diffusion/",transform=large_transform)
            np.random.seed(seed)

            subset_indices2 = np.random.choice(len(testsetout2), 2000, replace=False)
            subset_val_indices2 = subset_indices2[:int(0.2 * len(subset_indices2))]  
            subset_near_indices2 = subset_indices2[int(0.2 * len(subset_indices2)):]  
            subset_val2 = torch.utils.data.Subset(testsetout2, subset_val_indices2)
            sub_dataloader_dict_val = {'placesbg': torch.utils.data.DataLoader(subset_val, batch_size=batch_size, shuffle=False, num_workers=4),'stable_diffusion': torch.utils.data.DataLoader(subset_val2, batch_size=batch_size, shuffle=False, num_workers=4)}
            dataloader_dict['ood']['val']= sub_dataloader_dict_val

            subset_near = torch.utils.data.Subset(testsetout, subset_near_indices)
            subset_near2 = torch.utils.data.Subset(testsetout2, subset_near_indices2)
            sub_dataloader_dict_near = {'placesbg': torch.utils.data.DataLoader(subset_near, batch_size=batch_size, shuffle=False, num_workers=4),'stable_diffusion': torch.utils.data.DataLoader(subset_near2, batch_size=batch_size, shuffle=False, num_workers=4)}
            dataloader_dict['ood']['near'] = sub_dataloader_dict_near


            ood_datasets = [ 'gaussian', 'SVHN', 'iSUN', 'LSUN_resize', 'textures']

            dataloader_dict['ood'].setdefault('far', {})


            for dataset_name in ood_datasets:
                if dataset_name == "SVHN":
                    testsetout = svhn.SVHN(f"/home/hasani2/R/Datasets/OOD_Datasets/{dataset_name}", split='test',
                                        transform=large_transform, download=False)
                elif dataset_name == 'gaussian':
                    testsetout = GaussianDataset(dataset_size =10000, img_size = 224,
                        transform=transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
                else:
                    testsetout = ImageFolder(f"/home/hasani2/R/Datasets/OOD_Datasets/{dataset_name}", transform=large_transform)

                num_samples = len(testsetout)
                val_size = int(0.2 * num_samples)  
                np.random.seed(seed)
                val_indices = np.random.choice(num_samples, val_size, replace=False)
                np.random.seed(seed)
                far_indices = np.random.choice(list(set(range(num_samples)) - set(val_indices)), num_samples - val_size, replace=False)

                # Create the 'far' subset for this dataset (80% - val_size)
                subset_val = torch.utils.data.Subset(testsetout, val_indices)
                subset_far = torch.utils.data.Subset(testsetout, far_indices)
                sub_dataloader_dict = {}
                sub_dataloader_dict[dataset_name] = torch.utils.data.DataLoader(subset_far, batch_size=batch_size, shuffle=False,
                                                                                num_workers=4)
                dataloader_dict['ood']['far'].update(sub_dataloader_dict)
                sub_dataloader_dict_val[dataset_name] = torch.utils.data.DataLoader(subset_val, batch_size=batch_size, shuffle=False ,
                                                                                    num_workers=4)
                dataloader_dict['ood']['val'].update(sub_dataloader_dict_val)

        return dataloader_dict