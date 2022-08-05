<p align="center">
  <a href="https://github.com/dhyan1999/Hate_Speech_Detection" title="Hate Speech Detection">
  </a>
</p>
<h1 align="center"> A Comparative study of Data-Augmentation Techniques for Imbalanced Hate speech data </h1>

![uni](img/uni.jpeg)
Abstract : <p > Social media and microblogging apps allow people to share their information and express their personal view-points extensively and immediately. However, it also has some negative aspects such as hate speech. Recent advances in Natural Language Processing and Artificial Intelligence allow for more accurate detection of hate speech in textual streams. A significant challenge in this domain is that, while the presence of hate speech can be detrimental to the quality of service provided by social platforms, it still constitutes only a tiny fraction of the content available online, which can lead to performance deterioration due to majority class overfitting. To this end, we propose various data augmentation techniques with the goal of reducing class imbalance and maximizing the amount of information we can extract from our limited resources. After that, we apply them on a selection of top-performing deep architectures and hate speech datasets in order to classify them. The proposed approach outperforms all other considered algorithms. It achieves 0.69 F1-score for hate/non-hate classification</p>

<h2 align="center">🌐 Links 🌐</h2>
<p align="center">
    <a href="https://github.com/dhyan1999/Hate_Speech_Detection" title="Helmet Detection">📂 Repo</a>
    ·
    <a href="https://github.com/dhyan1999/Hate_Speech_Detection/blob/main/Report/Hate_Speech_Detection.pdf" title="Helmet Detection">📄 Paper</a>
    
</p>

<div>
    <a href="https://plotly.com/~dhyan1999/1/" target="_blank" title="Final Bert Augmentation Output" style="display: block; text-align: center;"><img src="https://plotly.com/~dhyan1999/1.png" alt="Final Bert Augmentation Output" style="max-width: 100%;width: 600px;"  width="600" onerror="this.onerror=null;this.src='https://plotly.com/404.png';" /></a>
</div>

## Table of Content

1. [Manifest](#-manifest)
2. [Prerequisites](#-prerequisites)
8. [Implementation of Code](##-implementation-of-code)
8. [Results](#-future-scope)
9. [Video](#video)

## 🧑🏻‍🏫 Manifest


```
- Code - Contains all parts of code in a sequential manner
- Dataset - Dataset that we have used in our project (Augmented Dataset as well)
- Presentation - Final Presentation
- Report - IEEE Paper for the project
```


## 🤔 Prerequisites

- [Python](https://www.python.org/ "Python") Installed

- Python Basics Understanding

- Understanding of Machine Learning and Deep Learning libraries

- Concepts of Natural Langauge Processing

## 👨🏻‍💻 Implementation of Code

BERT Contextual Embedding
- We assume an invariance that sentences are natural even if the words in the sentences are replaced with other words with paradigmatic relations.
- At the word places, we stochastically swap out words with others that a bidirectional language model predicts. There are many context-sensitive terms, but they are all acceptable for enhancing the original language


![BERTCon](img/BERTCon.png)
```py
import nlpaug.augmenter.word.context_word_embs as aug
augmenter = aug.ContextualWordEmbsAug(model_path='bert-base-uncased', action="insert")
def augmentMyData(df, augmenter, repetitions=1, samples=200):
    augmented_texts = []
    # select only the minority class samples
    spam_df = df[df['label'] == 1].reset_index(drop=True) # removes unecessary index column
    for i in tqdm(np.random.randint(0, len(spam_df), samples)):
        # generating 'n_samples' augmented texts
        for _ in range(repetitions):
            augmented_text = augmenter.augment(str(spam_df['Text'].iloc[i]))
            augmented_texts.append(augmented_text)
    
    data = {
        'label': 1,
        'Text': augmented_texts
    }
    aug_df = pd.DataFrame(data)
    df = shuffle(df.append(aug_df).reset_index(drop=True))
    return df
```

Assigning xml and mp4 file to variables

```py
cascade_src = 'bike.xml'
video_src = 'movie2.mp4'
```

Capture frames from a video

```py
cap = cv2.VideoCapture(video_src)
fgbg = cv2.createBackgroundSubtractorMOG2()
```

Trained XML classifiers describes some features of some object we want to detect

```py
car_cascade = cv2.CascadeClassifier(cascade_src)
```

Set up GUI

```py
window = tk.Tk()  #Makes main window
window.wm_title("Digital Microscope")
window.config(background="#FFFFFF")
```

Graphics window

```py
imageFrame = tk.Frame(window, width=600, height=500)
imageFrame.grid(row=0, column=0, padx=10, pady=2)
```

Capture video frames

```py
lmain = tk.Label(imageFrame)
lmain.grid(row=0, column=0)
```

```py
def show_frame():
	# reads frames from a video
    _, frame = cap.read()

    # convert to gray scale of each frames    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detects bikes of different sizes in the input image
    cars = car_cascade.detectMultiScale(gray, 1.59, 1)
```

To draw a rectangle in each bikes

```py
for (x, y, w, h) in cars:
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 215), 2)
    color = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    img = Image.fromarray(color)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(10, show_frame)
```

Slider window (slider controls stage position)

```py
sliderFrame = tk.Frame(window, width=600, height=100)
```
## 🎊 Future Scope

- In future this project can be extended to detect the number plates of bikes. 

- Proper GUI

## 🧑🏻 Author

**Dhyan Shah**

- 🌌 [Profile](https://github.com/dhyan1999 "Dhyan Shah")

- 🏮 [Email](mailto:dhyan.shah99@gmail.com?subject=Hi%20from%20Dhyan%20Shah "Hi!")

<p align="center">Made with Python & ❤️ in India</p>

</script>