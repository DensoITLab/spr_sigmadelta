echo "----run_exp----";
case $1 in
    "NM" )
        CONFIG="config/settings_mnist.yaml"
        kernel_mode='conv_conv'
        quantizer='MG_muldiv_log'
        MAC_loss=1
        expantion_ratio=1
        case $2 in 
            "0" )
                kernel_mode='conv_nan'
                MAC_loss=-1
                ;;
            "1" )
                expantion_ratio=1
                ;;           
            "2" )
                expantion_ratio=2
                ;;
            "3" )
                expantion_ratio=4
                ;;
            "4" )
                expantion_ratio=8
                ;;
            "5" )
                quantizer='ULSQ_muldiv_log'
                ;;
            "6" )
                MAC_loss=0
                ;;
            "7" )
                kernel_mode='conv_nan'
                ;;

        esac
        ;;
    "PI" )
        CONFIG="config/settings_pilotnet.yaml"
        kernel_mode='conv_conv'
        quantizer='MG_muldiv_log'
        MAC_loss=1
        expantion_ratio=1
        case $2 in  
            "0" )
                kernel_mode='conv_nan'
                MAC_loss=0
                ;;
            "1" )
                expantion_ratio=1
                ;;           
            "2" )
                expantion_ratio=2
                ;;
            "3" )
                expantion_ratio=4
                ;;
            "4" )
                expantion_ratio=8
                ;;
            "5" )
                quantizer='ULSQ_muldiv_log'
                ;;
            "6" )
                MAC_loss=0
                ;;
            "7" )
                kernel_mode='conv_nan'
                ;;
        esac
        ;;

    "NC" )
        CONFIG="config/settings_ncaltech.yaml"
        kernel_mode='conv_conv'
        quantizer='MG_muldiv_log'
        MAC_loss=1
        expantion_ratio=1
        case $2 in 
            "0" )
                kernel_mode='conv_nan'
                MAC_loss=0
                ;;
            "1" )
                expantion_ratio=1
                ;;      
            "2" )
                expantion_ratio=2
                ;;           
            "3" )
                quantizer='ULSQ_muldiv_log'
                ;;
            "4" )
                kernel_mode='conv_nan'
                ;;
        esac
        ;;

    "PS" )
        CONFIG="config/settings_prophesee.yaml"
        kernel_mode='conv_conv'
        quantizer='MG_muldiv_log'
        MAC_loss=1
        expantion_ratio=1
        case $2 in 
            "0" )
                kernel_mode='conv_nan'
                MAC_loss=0
                ;;
            "1" )
                expantion_ratio=1
                ;;    
            "2" )
                expantion_ratio=2
                ;; 
            "3" )
                quantizer='ULSQ_muldiv_log'
                ;;          
            "4" )
                kernel_mode='conv_nan'
                ;;
        esac
        ;;

esac
echo "Dataset: $1"
echo "Exe Idx: $2"
echo "CONFIG: $CONFIG"
echo "kernel_mode: $kernel_mode"
echo "quantizer: $quantizer"
echo "MAC_loss: $MAC_loss"
    echo "expantion_ratio: $expantion_ratio"
python3 train.py --settings_file $CONFIG --kernel_mode $kernel_mode  --MAC_loss $MAC_loss --expantion_ratio $expantion_ratio --quantizer $quantizer
echo "----done---";