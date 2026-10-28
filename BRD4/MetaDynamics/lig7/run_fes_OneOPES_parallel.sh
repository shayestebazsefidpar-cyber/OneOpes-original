
export input_dir="/mnt/FAMOB2/Projets2/dlukauskis/METAD_TIP3P/TIP3P/lig7/"

compute_fes() {
  cd replicate$i
  rm fes* delta*

  #Trypsin
  cp ../../funnel_FES_from_Reweighting.py .
  ./funnel_FES_from_Reweighting.py --skiprows 200000 --sigma 0.02 --bias metad.rbias --colvar "$input_dir/run$i/COLVAR" --cv pp.proj --bin 200 --temp 300 --min 1.54 --max 3.4 --rfunnel 0.25 --uat 3.0 --bat 1.8 --blocks 3 --outfile fes_blocks.dat
  ./funnel_FES_from_Reweighting.py --skiprows 200000 --sigma 0.02 --bias metad.rbias --colvar "$input_dir/run$i/COLVAR" --cv pp.proj --bin 200 --temp 300 --min 1.54 --max 3.4 --rfunnel 0.25 --uat 3.0 --bat 1.8 --outfile fesskip25k.dat
  ./funnel_FES_from_Reweighting.py --skiprows 200000 --sigma 0.02 --bias metad.rbias --colvar "$input_dir/run$i/COLVAR" --cv pp.proj --bin 200 --temp 300 --min 1.54 --max 3.4 --rfunnel 0.25 --uat 3.0 --bat 1.8 --stride 50000;  grep 'fundeltaF' fes-* | awk '{print $4}' > deltaF.dat

  #append all the deltaF in one file
  paste delta* > deleteme;

  #calculate average and stdev of all replicas in time
  #Trypsin
  awk '{sum = 0; for (i = 1; i <= NF; i++) sum += $i; sum /= NF; print NR*50000+200000, sum}' deleteme > temp1;
  awk '{sum = 0; sum2 = 0; for (i = 1; i <= NF; i++) sum += $i; sum /= NF; for (i = 1; i <= NF; i++) sum2 += ($i-sum)^2; sum2 /= NF; print sum2}' deleteme > temp2;

  paste temp1 temp2 > deltaFall.dat;
  rm temp* deleteme

  mkdir cmap_stride
  rm cmap_stride/fes*
  cd cmap_stride
  ../funnel_FES_from_Reweighting.py --skiprows 200000 --sigma 0.05 --bias metad.rbias --colvar "$input_dir/run$i/COLVAR" --cv cmap --bin 100 --temp 300 --min 0.0 --max 3.6  --stride 50000 --rfunnel 0.25 --uat 3.0 --bat 1.8
  cd ..


  ./funnel_FES_from_Reweighting.py --skiprows 200000 --sigma 0.01,0.05 --bias metad.rbias --colvar "$input_dir/run$i/COLVAR" --cv pp.proj,cmap --bin 400,100 --temp 300 --min 1.54,0.0 --max 3.4,3.6 --rfunnel 0.25 --uat 3.0 --bat 1.8 --outfile fess2D.dat


  cd ..

  echo "Replicate $i"
}

for i in {0..2}
do
  compute_fes &
done
wait

