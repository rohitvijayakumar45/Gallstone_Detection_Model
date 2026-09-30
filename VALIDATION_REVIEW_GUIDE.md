# 🖼️ Validation Image Review Tool

## 🚀 Quick Start

### **Run the Review Tool**
```bash
python scripts/review_validation_images.py
```

### **With Custom Dataset**
```bash
python scripts/review_validation_images.py --data_dir dataset_final_resplit
```

---

## 🎮 Controls

| Key | Action | Description |
|-----|--------|-------------|
| **← Left Arrow** or **A** | Previous Image | Go back to previous image |
| **→ Right Arrow** or **D** | Next Image | Move to next image |
| **SPACE** | Next Image | Move to next image |
| **B** | Mark as Bad | Mark current image as bad quality |
| **G** | Mark as Good | Mark current image as good quality |
| **X** or **DEL** | Delete Bad | Delete image marked as bad (requires B first) |
| **S** | Skip | Skip current image without marking |
| **Q** | Quit | Save progress and exit |
| **R** | Reset | Reset all reviews (requires confirmation) |

---

## 📊 What You'll See

### **Image Display**
- Original validation image with bounding boxes
- Green boxes = ground truth gallstone annotations
- Info overlay showing:
  - Image number (e.g., "Image 1/246")
  - Image filename
  - Number of boxes
  - Current status (GOOD/BAD)

### **Status Indicators**
- **GREEN text**: New image, not yet reviewed
- **YELLOW text**: Previously reviewed image
- **RED text**: Image marked as BAD

---

## 🎯 What to Look For

### **Mark as BAD (B) if:**
- ❌ Image is blurry or poor quality
- ❌ Gallstone not clearly visible
- ❌ Incorrect annotations (boxes don't match stones)
- ❌ Multiple stones but only one annotated
- ❌ No stone present but annotated
- ❌ Image corrupted or unreadable

### **Keep as GOOD (G) if:**
- ✅ Clear image quality
- ✅ Gallstone clearly visible
- ✅ Accurate annotations
- ✅ Proper lighting and contrast
- ✅ Representative of typical cases

---

## 💾 Auto-Save

- Progress is automatically saved every 10 images
- Final save when you quit (Q)
- Review state saved to: `dataset_final_resplit/review_state.json`

---

## 🗑️ Deleting Bad Images

### **Process:**
1. Press **B** to mark image as bad
2. Press **X** or **DEL** to delete the bad image
3. Both image and label file will be deleted

### **Safety:**
- Can only delete images marked as BAD
- Confirmation not required (be careful!)
- Deleted images are permanently removed

---

## 📈 Review Workflow

### **Recommended Approach:**

1. **Start Review**
   ```bash
   python scripts/review_validation_images.py
   ```

2. **Systematic Review**
   - Go through images sequentially
   - Mark bad images with **B**
   - Delete obviously bad ones with **D**
   - Skip uncertain ones with **S**

3. **Take Breaks**
   - Press **Q** to save and quit
   - Resume later - progress is saved
   - Continue from where you left off

4. **Complete Review**
   - Review all 246 validation images
   - Clean up bad quality images
   - Ensure high-quality validation set

---

## 🔍 After Review

### **Check Results**
```bash
# See review statistics
cat dataset_final_resplit/review_state.json

# Count remaining images
ls dataset_final_resplit/val/images/ | wc -l
```

### **Update Training**
After cleaning validation set:
```bash
# Re-run preprocessing with cleaned data
python train.py --phase 1 --data_dir ./dataset_final_resplit --preprocess-limit -1

# Train model with clean validation set
python train.py --phase 3 --data_dir ./dataset_final_resplit --train-yolo \
    --yolo-weights yolo26m.pt --epochs 150 --batch-size 8
```

---

## 📊 Expected Results

### **Before Review:**
- Validation images: 246
- Mixed quality
- Potential labeling errors

### **After Review:**
- Validation images: ~200-230 (estimated)
- High quality only
- Accurate labels
- Better model evaluation

---

## 🚨 Important Notes

1. **Backup First**: Consider backing up original dataset before deleting
2. **Be Conservative**: When in doubt, keep the image (press S)
3. **Focus on Quality**: Remove obviously bad images first
4. **Take Your Time**: Quality review is worth the effort
5. **Save Often**: Progress is auto-saved, but quit with Q to be safe

---

## 🎯 Benefits of Clean Validation Set

### **Better Model Evaluation:**
- ✅ More accurate mAP50 measurements
- ✅ Reliable performance metrics
- ✅ Better hyperparameter optimization
- ✅ Trustworthy model comparisons

### **Improved Training:**
- ✅ Better generalization estimates
- ✅ More realistic performance expectations
- ✅ Easier debugging of model issues
- ✅ Clearer understanding of model limitations

---

## 🚀 Ready to Review!

**Start the review tool:**
```bash
python scripts/review_validation_images.py
```

**Tips:**
- Set aside dedicated time for review (30-60 minutes)
- Use a large monitor for better visibility
- Take breaks to maintain focus
- Be systematic in your approach

**Good luck cleaning your validation set!** 🎯