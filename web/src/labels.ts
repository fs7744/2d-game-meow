/**
 * 显示层中文映射表。
 *
 * ## 为什么要有这张表
 *
 * 界面上的这些英文（`Naked` / `angry 😠` / `Shonen Anime` / `Ballet`）
 * **不是文案，是数据标识** —— 它们同时是磁盘目录名、后端接口的键、
 * 以及要写进 prompt 的内容。所以**一个都不能改**：
 *
 * | 标识 | 落地成什么 | 改了会怎样 |
 * |---|---|---|
 * | 服装名 `Naked` | `assets/characters/<角色>/Sprites/Naked/` | 素材找不到 |
 * | 表情 `safe_name` | 目录名 + 提交给后端的参数 | 后端认不出 |
 * | 画风 `id` | `character_styles.json` 的键 | 选中后取不到 prompt |
 * | 姿势分类/名字 | `PoseLibrary/` 的目录与文件名 | 筛不出姿势 |
 * | 属性 `tag` | **写进 prompt 的实际内容** | 出图直接变样 |
 *
 * 所以做法是**只翻译显示层**：值一律传原样的英文，界面上显示中文。
 *
 * 查不到译文时一律回落到原值（`zh()`），**绝不显示空白也不显示 undefined** ——
 * 用户自建的服装名、自存的姿势名本来就该原样显示。
 */

/** 取译文：查不到就退回原值（用户自己起的名字就该原样显示）。 */
export function zh(map: Record<string, string>, key: string): string {
  return map[key] ?? key
}

/* ------------------------------------------------------------------ *
 * 服装名（官方 `character_cloner.py` 产出的基础集 + 我们自己的白内衣）
 * ------------------------------------------------------------------ */

/** 基础服装集的中文名。用户自建的服装不在表里，原样显示。 */
export const COSTUME_LABEL: Record<string, string> = {
  Naked: '裸装',
  Original: '原始',
  Underwear: '白内衣',
}

/* ------------------------------------------------------------------ *
 * 背景色（官方 `background_color`，抠图按它取样）
 * ------------------------------------------------------------------ */

/** 幕布色中文名。译成「绿幕」而不是「绿色」—— 抠图语境里是幕布。 */
export const BG_COLOR_LABEL: Record<string, string> = {
  green: '绿幕',
  blue: '蓝幕',
  white: '白幕',
  grey: '灰幕',
  magenta: '品红幕',
  transparent: '透明',
}

/* ------------------------------------------------------------------ *
 * 表情（官方 `emotions.json`，157 条）
 * ------------------------------------------------------------------ */

/**
 * 表情分组中文名（`emotions.json` 的顶层 key）。
 */
export const EMOTION_CATEGORY_LABEL: Record<string, string> = {
  'Anger 😠': '愤怒',
  'Smile 😊': '微笑',
  'Smug 😏': '得意',
  'Surprise 😲': '惊讶',
  'Emotional 😢': '感伤',
  'Sexual 🔥': '情欲',
  'Focus/Thought 🤔': '专注',
  'Pain/Ailment 🤕': '痛苦',
  'Emotes 🤪': '颜文字',
}

/**
 * 表情中文名，键是 `safe_name`（唯一、稳定、不含 emoji）。
 *
 * emoji **不写在这里** —— 原 `key` 里已经有，从那儿原样取走，
 * 免得手抄 emoji 抄错（`😮‍💨` 这种带变体选择符的，一个字符都不止一个码位）。
 */
export const EMOTION_LABEL: Record<string, string> = {
  /**
   * `neutral` 不在官方 157 条 `safe_name` 里（官方表情库从 angry 开始），
   * 但它是**我们这边的默认表情**（后端 `EMOTIONS = ["neutral"]`，
   * 磁盘目录 `Sprites/<服装>/neutral/`）—— 界面上一定会出现，不能漏。
   */
  neutral: '平静',
  Neutral: '平静',

  /* 愤怒 */
  angry: '愤怒',
  rage: '暴怒',
  annoyed: '不耐烦',
  frustrated: '沮丧',
  scowl: '皱眉怒视',
  'boiling-rage': '怒火中烧',
  'cold-anger': '冰冷的怒意',
  disdain: '鄙夷',
  'gritted-teeth': '咬牙切齿',
  'furious-snarl': '怒容咆哮',
  seething: '暗涌的怒火',
  'explosive-fury': '爆发的怒火',
  'glare-of-judgment': '审判的目光',
  'trembling-rage': '颤抖的愤怒',
  'angry-pout-2': '生气地撅嘴',
  'veins-pop': '青筋暴起',
  'scream-of-fury': '愤怒尖叫',
  'evil-cackle': '邪恶的坏笑',

  /* 微笑 */
  smile: '微笑',
  grin: '露齿笑',
  laughing: '大笑',
  'light-smile': '浅笑',
  'radiant-smile': '灿烂微笑',
  'shy-smile': '羞涩微笑',
  'mischievous-grin': '淘气坏笑',
  'content-smile': '满足微笑',
  'smile-dimples': '笑出酒窝',
  'gentle-grin': '温和微笑',
  'sweet-smile': '甜美微笑',
  'cat-smile-nyan': '猫猫笑',

  /* 得意 */
  smirk: '坏笑',
  smug: '得意',
  'troll-face': '恶作剧脸',
  doyagao: '摆烂脸',
  'victorious-smirk': '胜利者的坏笑',
  'mocking-smile': '嘲弄的笑',
  'knowing-grin': '会心的坏笑',
  'sly-smile': '狡黠的笑',
  'condescending-smirk': '居高临下的冷笑',
  'tsundere-pout': '傲娇撅嘴',
  'yandere-smile': '病娇微笑',
  'wry-smile': '苦笑',

  /* 惊讶 */
  surprised: '惊讶',
  shocked: '震惊',
  traumatized: '精神受创',
  'turn-pale': '脸色发白',
  'delighted-surprise': '又惊又喜',
  'shocked-silence': '震惊到沉默',
  'sudden-realization': '恍然大悟',
  'startled-blink': '受惊地眨眼',
  'jaw-drop': '目瞪口呆',
  'sudden-shock': '猝然受惊',
  'awe-reverence': '敬畏',

  /* 感伤 */
  sad: '难过',
  crying: '哭泣',
  depressed: '低落',
  despair: '绝望',
  sobbing: '抽泣',
  'nostalgic-sadness': '怀旧的伤感',
  bittersweet: '苦乐参半',
  'emotional-vulnerability': '情感脆弱',
  'silent-tears': '默默流泪',
  'aching-heart': '心痛',
  'silent-scream-anguish': '无声的尖叫',
  downtrodden: '意志消沉',
  'exhausted-sigh': '疲惫叹息',

  /* 情欲 */
  'seductive-smile': '妩媚一笑',
  aroused: '情动',
  'in-heat': '发情',
  afterglow: '余韵',
  torogao: '娇喘',
  'naughty-face': '淘气脸',
  ohhoai: '气喘吁吁',
  ahegao: '情迷失神',
  'teasing-glance': '挑逗的眼神',
  'dominant-allure': '强势的诱惑',
  'submissive-blush': '顺从的红晕',
  'sensual-tension': '感官张力',
  'bedroom-eyes': '迷离眼神',
  'soft-moan': '轻吟',
  'biting-lip': '咬唇',
  'tongue-tease': '吐舌挑逗',
  'flushed-desire': '情欲泛红',
  'sultry-pout': '性感撅嘴',
  'erotic-daze': '情欲恍惚',
  'tongue-out-sexual': '吐舌',
  'moaning-face': '呻吟脸',
  'glazed-eyes-sexual': '迷离失神的眼',
  'choking-face-strain': '窒息挣扎',
  'kissing-face': '亲吻脸',

  /* 专注 */
  concentration: '专注',
  meditation: '冥想',
  'deep-thought': '沉思',
  unimpressed: '不以为然',
  'deadpan-face': '面无表情',
  bored: '无聊',
  skeptical: '怀疑',
  'cynical-face': '讥讽',
  'side-eye': '白眼',
  puzzled: '困惑',
  'amusement-disbelief': '哭笑不得',

  /* 痛苦 */
  'physical-pain': '身体疼痛',
  headache: '头痛',
  wincing: '忍痛',
  'biting-lower-lip': '咬住下唇',
  'confused-perplexity': '迷惑',

  /* 颜文字 */
  wink: '眨眼',
  'cheeky-grin': '调皮笑',
  'laughing-face': '大笑',
  'shock-face': '震惊',
  'cat-smile': '猫笑',
  'flirty-cat': '调情猫',
  'anime-grin': '动漫坏笑',
  'pout-face': '撅嘴',
  'excited-anime': '兴奋',
  uwu: '撒娇',
  'tongue-out': '吐舌',
  'wink-tongue': '眨眼吐舌',
  'quirky-face': '古怪表情',
  'flirty-pucker': '调情嘟嘴',
  'evil-grin': '邪恶坏笑',
  'angry-emote': '生气',
  'awkward-face': '尴尬',
  indifferent: '漠然',
  uncertain: '不确定',
  'neutral-emote': '中性',
  'x-mouth': '闭嘴',
  'sad-emote': '难过',
  'hopeful-smile': '满怀希望的笑',
  disappointed: '失望',
  'winking-frown': '眨眼皱眉',
  'diamond-mouth': '菱形嘴',
  'smug-smile': '得意笑',
  'awkward-grin': '尴尬笑',
  'surprised-emote': '惊讶',
  'winking-surprise': '眨眼惊讶',
  'anime-pout': '动漫撅嘴',
  'scrunched-pout': '皱脸撅嘴',
  'blank-stare': '呆滞凝视',
  'anime-shock': '动漫震惊',
  expressionless: '无表情',
  'tired-face': '疲惫',
  'circle-eyes': '黑豆眼',
  'heart-eyes': '爱心眼',
  'happy-emote': '开心',
  cheering: '欢呼',
  'playful-pucker': '俏皮嘟嘴',
  'frustrated-emote': '沮丧',
  'angry-pout': '生气撅嘴',
  dizzy: '头晕',
  suspicious: '警惕',
  exhausted: '精疲力尽',
  sarcastic: '讽刺',
  deadpan: '面瘫',
  'crying-emote': '哭',
  'anime-glare': '动漫怒视',
  'focused-glare': '专注怒视',
}

/**
 * 表情显示名 = 中文名 + 原 emoji。
 *
 * emoji 从原 `key` 末尾摘（`angry 😠` → `😠`），而不是写死在表里 ——
 * `😮‍💨` / `😵‍💫` 这类带变体选择符（U+FE0F）和零宽连接符（U+200D），
 * 手抄几乎必错。
 */
export function emotionLabel(safeName: string, key: string): string {
  const cn = EMOTION_LABEL[safeName]
  if (!cn) return key
  const m = key.match(/(?:\p{Extended_Pictographic}|\p{Emoji_Component}|\uFE0F|\u200D)+\s*$/u)
  return m ? `${cn} ${m[0].trim()}` : cn
}

/* ------------------------------------------------------------------ *
 * 画风（官方 `character_styles.json`，13 组 358 条）
 * ------------------------------------------------------------------ */

/**
 * 画风分组中文名。
 *
 * 键是 `character_styles.json` 的顶层 `label`（不是 id）——
 * 界面上显示的就是它，按它翻译最省一层映射。
 */
export const STYLE_GROUP_LABEL: Record<string, string> = {
  'Anime Meta Styles': '动漫元风格',
  Anime: '动漫',
  Animation: '动画',
  Artistic: '艺术',
  Realistic: '写实',
  Cartoon: '卡通',
  Comics: '漫画',
  Drawing: '绘画',
  Design: '设计',
  '3D Render': '3D 渲染',
  'Digital Painting': '数字绘画',
  'Cover Art': '封面插画',
  Painting: '传统绘画',
}

/**
 * 画风中文名，键是 `label`（358 条互不重复，已校验）。
 *
 * **作者名一律音译成通行中文译名**（宫崎骏、大友克洋、浦安直人…），
 * 实在没有通行译名的画师保留英文原名（`Shindol`、`Sakimichan` 等）——
 * 音译错的专名比英文更难认。
 */
export const STYLE_LABEL: Record<string, string> = {
  /* ---- Anime Meta Styles ---- */
  'Shonen Anime': '少年漫',
  'Shojo Anime': '少女漫',
  'Seinen Anime': '青年漫',
  'Josei Anime': '女性向漫',
  '1970s Anime': '70 年代动画',
  '1980s Anime': '80 年代动画',
  '1990s Anime': '90 年代动画',
  '2000s Anime': '00 年代动画',
  '2010s Anime': '10 年代动画',
  '2020s Anime': '20 年代动画',

  /* ---- Anime ---- */
  'Hayao Miyazaki / Studio Ghibli': '宫崎骏 / 吉卜力',
  'Makoto Shinkai': '新海诚',
  'Kyoto Animation / Violet Evergarden': '京都动画 /《紫罗兰永恒花园》',
  'Kyoto Animation / K-On!': '京都动画 /《轻音少女》',
  'Hiroyuki Imaishi / Studio Trigger': '今石洋之 / Trigger',
  'Yoshiyuki Sadamoto': '贞本裕行',
  CLAMP: 'CLAMP 组合',
  'Sailor Moon / 1990s Anime': '《美少女战士》/ 90 年代动画',
  'Akira Toriyama': '鸟山明',
  'Rumiko Takahashi': '高桥留美子',
  'Katsuhiro Otomo': '大友克洋',
  'Satoshi Kon': '今敏',
  'Ghost in the Shell / 1995 Film': '《攻壳机动队》/ 1995 年剧场版',
  'Toshihiro Kawamoto': '川元利弘',
  'Yoshitoshi ABe': '阿部悟司',
  'Nobuteru Yuki': '冲良和',
  'Anime Style': '动漫风',
  'Manga Style': '漫画风',
  'Atsushi Ohkubo Style': '大久保笃风格',
  'Attack on Titan Style': '《进击的巨人》风格',
  'Chainsaw Man Style': '《电锯人》风格',
  'Jujutsu Kaisen Style': '《咒术回战》风格',
  'My Hero Academia Style': '《我的英雄学院》风格',
  'Berserk Manga Style': '《剑风传奇》漫画风',
  'Bleach Style': '《死神》风格',
  'Frieren Style': '《葬送的芙莉莲》风格',
  'Boichi Style': '漩涡一雄风格',
  'Cinematic Anime': '电影感动画',
  'Classic Shojo Illustration': '经典少女漫画插画',
  'Euromanga Style': '欧式漫画风格',
  'Watercolor Anime Illustration': '水彩动画插画',
  'Granblue Fantasy Pin-Up Style': '《碧蓝幻想》海报风',
  'Katsuya Terada Style': '寺田胜也风格',
  'Manhwa Ultradetailed Art': '韩漫超精细画风',
  'Murata Hyperrealism': '村田雄飞式超写实',
  'MidJourney Niji-Inspired Digital Illustration': 'MidJourney Niji 风格数字插画',
  'Shindol Style': 'Shindol 风格',
  'Solo Leveling Art': '《我独自升级》画风',
  'Pony 2.5D Realism': 'Pony 2.5D 写实',
  'Pony/Illustrious Checkpoint Style': 'Pony / Illustrious 底模风格',
  'Yoshitaka Amano Style': '天野喜孝风格',
  'Yoshitaka Amano Fine Art': '天野喜孝精致画风',
  'Death Note Style': '《死亡笔记》风格',
  "JoJo's Bizarre Adventure Style": '《JOJO的奇妙冒险》风格',
  'One Piece Style': '《海贼王》风格',
  'Chibi Anime': 'Q 版动漫',
  'Anime Key Visual': '动画主视觉',
  'Junji Ito Style': '伊藤润二风格',
  'Anime Illustration': '动漫插画',
  'Ultradetailed Manga Illustration': '超精细漫画插画',
  'Polished Ultrarealistic Anime Pin-Up': '精致超写实动漫海报',
  'Thin Delicate Outline Romantic Anime-Photographic Hybrid':
    '纤细线条浪漫动漫×摄影混合',
  'Cel Shading': '赛璐璐上色',
  'Khyleri Style': 'Khyleri 风格',
  'Modern Pokémon Anime Style': '现代宝可梦动画风格',
  'Naruto Style': '《火影忍者》风格',
  'Polished Mobile Gacha Style': '精致手游画风',
  'Crayon Shin-chan Style': '《蜡笔小新》风格',
  'Mecha Anime Style': '机甲动漫风格',
  'Osamu Tezuka Style': '手冢治虫风格',
  'Yu-Gi-Oh! Style': '《游戏王》风格',
  'Satoshi Urushihara Style': '浦河直人风格',
  'Naoki Urasawa Style': '浦安直人风格',
  'Hiroyuki Okiura Style': '押井守风格',
  'Attack on Titan (WIT Studio) Style': '《进击的巨人》(WIT 制作) 风格',
  'Attack on Titan (MAPPA) Style': '《进击的巨人》(MAPPA 制作) 风格',
  'Gainax Style': 'GAINAX 风格',

  /* ---- Animation ---- */
  'Disney Renaissance': '迪士尼复兴期',
  'Don Bluth': '顿·布卢特',
  'Bruce Timm / DCAU': '布鲁斯·蒂姆 / 迪士尼动画宇宙',
  'Genndy Tartakovsky': '根尼·塔塔科夫斯基',
  'Cartoon Saloon': 'Cartoon Saloon 工作室',
  'Disney / Tangled': '迪士尼 /《长发公主》',
  'Pixar / The Incredibles': '皮克斯 /《超人总动员》',
  'Fortiche / Arcane': 'Fortiche /《双城之战》',
  'Spider-Verse': '《蜘蛛侠：平行宇宙》',
  'Henry Selick / Coraline': '亨利·塞利克 /《鬼妈妈》',

  /* ---- Artistic ---- */
  'Art Nouveau / Alphonse Mucha': '新艺术运动 / 阿方斯·穆夏',
  'Art Deco / Tamara de Lempicka': '装饰艺术 / 塔玛拉·德·伦皮卡',

  /* ---- Realistic ---- */
  'Academic Realism': '学院派写实',
  Photorealism: '照片写实',

  /* ---- Cartoon ---- */
  'Cartoon Style': '卡通风格',
  'Invader Zim Style': '《星际牛仔 Zim》风格',
  'Nickelodeon Animation Style': '尼克频道动画风格',
  'Rick and Morty Style': '《瑞克和莫蒂》风格',
  'Simpsons Style': '《辛普森一家》风格',
  'South Park Style': '《南方公园》风格',
  'Steven Universe Style': '《Steven Universe》风格',
  'Adventure Time Style': '《探险活宝》风格',
  'Hanna-Barbera Style': '汉纳-巴伯拉风格',
  'Family Guy Style': '《恶搞一家人》风格',
  'Modern Cartoon': '现代卡通',
  'Rubber Hose Cartoon': '橡皮软管卡通',
  'Disney Ultradetailed Illustration': '迪士尼超精细插画',
  'Classic Disney Feature Animation': '迪士尼经典长片动画',
  'Cyberpunk: Edgerunners Style': '《赛博朋克：边缘行者》风格',
  'Tex Avery Style': '泰克斯·艾弗里风格',
  'Gravity Falls Style': '《怪诞集》风格',
  'Futurama Style': '《飞出个未来》风格',
  'Peanuts Cartoon Style': '《花生漫画》风格',
  'X-Men: The Animated Series Style': '《X战记》动画风格',
  "Dexter's Laboratory Style": '《德克实验室》风格',
  'Powerpuff Girls Style': '《飞天小女警》风格',

  /* ---- Comics ---- */
  'Comics Style': '漫画风格',
  'Alex Ross Style': '亚历克斯·罗斯风格',
  'Alternative Comics Style': '另类漫画风格',
  'Amanda Conner Illustration': '阿曼达·康纳插画',
  'American Comic Realism': '美式漫画写实',
  'Comic Européen Prestige': '欧洲精品漫画',
  'American Pulp Illustration': '美式通俗小说插画',
  'Frank Miller Style': '弗兰克·米勒风格',
  'Bengal Illustration': '孟加拉插画',
  'Crosshatched Graphic Novel': '交叉排线图像小说',
  'Graphic Novel': '图像小说',
  'Franco-Belgian Ultradetailed Comics': '法比超精细漫画',
  'Frank Cho Style': '弗兰克·赵风格',
  'Enki Bilal-Inspired Comic and Digital Painting': '恩基·比拉尔风格漫画与数字绘画',
  'Hellboy Style': '《地狱男爵》风格',
  'Archie Comics Style': '阿奇漫画风格',
  'Bande Dessinée (Hergé/Goscinny Tradition)': '法式漫画（埃尔热 / 戈西尼传统）',
  'Mark Brooks / Manapul Style': '马克·布鲁克斯 / 曼纳普尔风格',
  'Marvel Comics Style': '漫威漫画风格',
  'Marvel Cover Art': '漫威封面',
  'Milo Manara Style': '米洛·马纳拉风格',
  'Psychological Thriller Pulp Digital Painting': '心理惊悚 pulp 数字绘画',
  'DC Comics Style': 'DC 漫画风格',
  'Zenescope Illustration': 'Zenescope 插画',
  'Ryan Benjamin Style': '瑞恩·本杰明风格',
  'Travis Charest Style': '特拉维斯·查瑞斯特风格',
  'Adi Granov Style': '阿迪·格拉诺夫风格',
  'Bold Contour Style': '粗犷轮廓风格',
  'Clean Line Art': '干净线稿',
  'Moebius-Inspired Illustration': '墨比乌斯风格插画',
  'Vintage Comics': '复古漫画',
  'Jim Lee Style': '吉姆·李风格',
  'Joe Quesada Style': '乔·克萨达风格',
  'Glossy Comic Digital Art': '光泽漫画数字艺术',
  'Photorealistic Comic Ink-and-Paint Hybrid': '照片写实漫画墨绘混合',

  /* ---- Drawing ---- */
  Drawing: '素描',
  Caricature: '漫画夸张画',
  'Charcoal Rendering': '木炭渲染',
  'Hyperrealistic Colored-Pen Sketch': '超写实彩铅素描',
  'Jarek Kubicki Style': '雅雷克·库比茨基风格',
  'Playful Hand-Drawn Illustration': '俏皮手绘插画',
  'Chinese Ink Drawing': '中国水墨画',
  Sketch: '速写',
  "Children's Book Drawing": '童书插画',
  "Vintage Children's Book Drawing": '复古童书插画',
  'Gris Grimly Style': '格里斯·格里姆利风格',
  'Tim Burton Style': '蒂姆·伯顿风格',
  'Fashion Sketch': '时装速写',
  'Loose Crosshatched Character Sketch': '松散排线人物速写',
  'Wojak/Meme-Sketch Style': 'Wojak / 梗图速写风格',
  'Ink Pen Drawing': '钢笔淡墨画',
  Etching: '蚀刻版画',
  'Technical Drawing': '技术制图',
  'Conté Pastel': '炭笔粉彩',
  'Crayola Colored-Book Style': '蜡笔画本风格',
  'Bookplate Illustration': '藏书票插画',
  'Boulet Graphic Illustration': '布勒平面插画',
  'Poe-Inspired Illustration': '爱伦坡风格插画',
  'Collector Storybook Illustration': '收藏版故事书插画',
  'Saul Steinbeck Art': '索尔·斯坦贝克艺术',
  'Whimsical Illustration': '奇幻插画',
  Doodle: '涂鸦',
  'Very Simplistic Doodle': '极简涂鸦',
  'Beatrix Potter Style': '碧雅翠丝·波特风格',
  'Lo-Fi Art': 'Lo-Fi 艺术',
  'Pastel-Toned Cute Illustration (Western-Korean Hybrid)': '粉彩色可爱插画（西韩混合）',
  'Glen Keane Style': '格伦·基恩风格',
  'Calligraphic Linework': '书法线条',
  'Traditional Marker Illustration': '传统马克笔插画',
  'Western Cel-Shaded Illustration': '欧美赛璐璐插画',
  'Mad Magazine Cover Art': '《MAD》杂志封面风',

  /* ---- Design ---- */
  Design: '设计',
  'Butcher Billy Style': '屠夫比利风格',
  'Corporate Memphis': '企业孟菲斯',
  '8-Bit Style': '8 比特风格',
  'Kintsugi-Inspired Digital Art': '金继风格数字艺术',
  'Soviet Propaganda Poster': '苏联宣传海报',
  'Minecraft Style': '《我的世界》风格',
  'Funko Pop Style': 'Funko Pop 风格',
  'Hyper-Stylized Illustration': '夸张风格插画',
  'Graphic Flat-Vector Travel Poster Illustration': '平面矢量旅行海报插画',
  'Ornamental Patterns Hyperrealism Portrait': '装饰纹样超写实肖像',
  'Flat Illustration': '扁平插画',
  'Vector Art': '矢量艺术',
  Bauhaus: '包豪斯',
  'Poster Art': '海报艺术',
  'Minimalist Line Art': '极简线稿',
  Minimalism: '极简主义',
  Gradients: '渐变',
  Origami: '折纸',

  /* ---- 3D Render ---- */
  '3D Render': '3D 渲染',
  'Pixar Animation': '皮克斯动画',
  'Cinematic Final Fantasy 3D Render': '电影级最终幻想 3D 渲染',
  'Marble Sculpture': '大理石雕塑',
  'Playful Retro Pop': '俏皮复古波普',
  'Low Poly': '低多边形',
  'LEGO Minifigure': '乐高小人',
  'LEGO Set Style': '乐高套装风格',
  'Paper Cutout': '剪纸',
  Collage: '拼贴',
  Claymation: '黏土定格动画',
  'Mixed Media': '混合媒介',
  'Wallace and Gromit Style': '《超级无敌掌门狗》风格',
  'Muppets Style': '《木偶秀》风格',
  'PS1 Graphics': 'PS1 画质',
  Woodcut: '木刻版画',
  'Contemporary Lithography': '现代石版画',
  'Variable-Size Ben-Day Dots Illustration': '变尺寸网点插画',
  'Contemporary Ukiyo-E': '现代浮世绘',
  'Mosaic Art': '马赛克艺术',
  'Stained Glass Flat Art': '彩绘玻璃平面艺术',
  Quilling: '衍纸工艺',
  'Lowbrow Style': '低眉风格',
  'Fortnite Style': '《堡垒之夜》风格',
  'Kurzgesagt Style': 'Kurzgesagt 风格',
  'Modern Super Mario (Nintendo) Style': '现代超级马里奥（任天堂）风格',
  'Star Wars: The Clone Wars Style': '《星球大战：克隆人战争》风格',
  'M.C. Escher Style': '埃舍尔风格',
  'Vignette-Framed Central Composition': '暗角框中央构图',
  '1960s Psychedelic Poster Art': '60 年代迷幻海报艺术',
  'Frutiger Aero': 'Frutiger Aero 风格',
  'Alex Grey Style': '亚历克斯·格雷风格',
  'Vignette Tattoo Design (Flat Print Format)': '暗角纹身设计（平面印刷）',
  'Vignette Sticker': '暗角贴纸',
  'Icon/Emoticon Vignette': '图标 / 表情符号暗角',
  'Bold Painterly Impasto': '大胆厚涂',
  'The Mitchells vs. The Machines animation Style': '《 Mitchell 一家大战机器人》动画风格',
  Steampunk: '蒸汽朋克',
  'Unreal Engine': '虚幻引擎',
  'Jean-Gabriel Domergue Style': '让-加布里埃尔·多梅尔格风格',
  'Neon-Based Drawing': '霓虹素描',
  'Op-Art Style': '欧普艺术风格',
  'Overwatch Concept Art': '《守望先锋》概念艺术',
  'Psychedelia Style': '迷幻风格',
  Playmobil: 'Playmobil 积木人',
  'Camille Walala Style': '卡米尔·瓦拉拉风格',
  'C.F.A. Voysey illustration': 'C.F.A. 沃西插画',
  'Neon Halftone Retro-Thriller Illustration': '霓虹网点复古惊悚插画',
  'Topographic Linework Illustration': '等高线插画',
  'Negative Space Focused Illustration': '负空间插画',
  'Collectible Figure': '收藏手办',
  'Tessellation-Based Illustration': '镶嵌图案插画',
  'Neon Sign Design': '霓虹灯牌设计',
  'Plastic Arts Sculpture': '塑料艺术雕塑',
  'PS1 Screenshot': 'PS1 截图',
  'Low Relief Sculpture': '低浮雕雕塑',
  'The Sims Game Screenshot': '《模拟人生》游戏截图',
  'Animal Crossing Style': '《动物森友会》风格',
  'Partial Wireframe 3D': '局部线框 3D',
  'Dead or Alive Style': '《死或生》风格',
  'Octopath Traveler HD-2D Style': '《八方旅人》HD-2D 风格',
  'Nintendo 64 Style': 'N64 风格',
  'Street Fighter IV Style': '《街头霸王 IV》风格',

  /* ---- Digital Painting ---- */
  'Digital Painting': '数字绘画',
  'Castlevania Concept Art': '《恶魔城》概念图',
  'Sakimichan Style': 'Sakimichan 风格',
  'Scribble Fine Art': '涂鸦精细艺术',
  'Gris Grimly-Inspired Hyperrealism': '格里斯·格里姆利式超写实',
  'Absolute Painterly Style': '极致厚涂风',
  'Artgerm Style': 'Artgerm 风格',
  'ArtStation Masterpiece': 'ArtStation 大师级作品',
  'Bastien Lecouffe-Deharme Fantasy': '巴斯蒂安·勒库夫-德阿姆奇幻',
  'Bernie Wrightson Gothic': '伯尼·莱特森哥特风',
  'Character Design Illustration': '角色设计插画',
  'Charlie Bowater Style': '查理·博瓦特风格',
  'Contemporary Fantasy Art': '当代奇幻艺术',
  'Cosmic Horror Illustration': '宇宙恐怖插画',
  'Contemporary Commercial Illustration': '当代商业插画',
  'Editorial Illustration': '编辑部插画',
  'Hades Style': '《哈迪斯》风格',

  /* ---- Cover Art ---- */
  'Cover Art': '封面插画',
  'Interplay of Shadow and Light': '光影交错',
  'Kekai Kotaki Style': 'Kekai Kotaki 风格',
  'Robin Eley Style': '罗宾·埃利风格',
  'Dark Fantasy Illustration': '黑暗奇幻插画',
  'GTA V Polished Cover Art': '《GTA V》精致封面',
  'Expressive Character Illustration': '富表现力的角色插画',
  'Splatterpunk Horror Illustration': '泼墨朋克恐怖插画',
  'James Jean Style': '詹姆斯·简风格',
  'Victo Ngai Style': 'Victo Ngai 风格',
  'Kay Nielsen Style': '凯·尼尔森风格',
  'Hyper-Stylized Digital Painting': '夸张数字绘画',
  'Movie Digital Art': '电影数字艺术',
  'RossDraws Style': 'RossDraws 风格',
  'Loish Ultrarealism': 'Loish 超写实',
  'J.C. Leyendecker Style': 'J.C. 莱登贝克风格',
  'Ismail Inceoglu Style': '伊斯梅尔·因杰奥卢风格',
  'Faceted Painterly Digital painting': '切面厚涂数字绘画',
  'Sharp-Lined Glossy Semi-Realistic Portrait': '锐线光泽半写实肖像',
  'Aleksi Briclot and Sparth-Inspired Art': '亚历克西·布里克洛与 Sparth 风格',
  'Marvel Rivals Character Design (Painterly Comic Illustration)':
    '《漫威争锋》角色设计（厚涂漫画插画）',
  'Crisp Semi-Realistic Anime Illustration': '清爽半写实动漫插画',
  'Glossy Semi-Realistic': '光泽半写实',
  'Ilya Kuvshinov Style': '伊利亚·库夫希诺夫风格',
  'Luminous Pastel': '柔光粉彩',
  'Melancholic Hyperrealism': '忧郁超写实',
  'MTG (Magic: The Gathering) Style': '《万智牌》风格',
  'Tom Bagshaw-Inspired Digital Painting': '汤姆·巴格肖风格数字绘画',
  'Action Movie Poster with Diagonal Speed Strokes': '斜向速度线动作片海报',

  /* ---- Painting ---- */
  Painting: '绘画',
  'Abstract Expressionism': '抽象表现主义',
  'Acrylic Dreamscape': '亚克力梦景',
  'Adventure Illustration': '冒险插画',
  'Agnes Cecile Style': '阿涅斯·塞西尔风格',
  'Airbrush Fantasy': '喷绘幻想',
  'Van Gogh Style': '梵高风格',
  'Monet Style': '莫奈风格',
  'Wet-on-Wet Watercolor Ink Painting': '湿画法水彩墨画',
  'Alphonse Mucha Masterwork': '穆哈杰作',
  'Anato Finnstark Style': '阿纳托·芬斯塔克风格',
  'Rembrandt Painting': '伦勃朗油画',
  'Art Deco Gold Highlights': '装饰艺术金色高光',
  'Ashley Wood Illustration': '阿什利·伍德插画',
  Tenebrism: '暗色调主义',
  Chiaroscuro: '明暗对照法',
  'Bill Sienkiewicz Expressionism': '比尔·先克维奇表现主义',
  'Graffiti Painting': '涂鸦绘画',
  'Boris Vallejo Style': '鲍里斯·巴列霍风格',
  'Brushwork Emphasis': '强调笔触',
  'Pin-Up': '海报女郎',
  '2020s Pin-Up': '20 年代海报女郎',
  Hyperrealism: '超写实',
  Ultrarealism: '极致写实',
  'Casey Baugh Style': '凯西·博风格',
  'Commercial Illustration': '商业插画',
  'Giallo Poster Art': '铅黄电影海报',
  'Grindhouse Poster Illustration': '磨坊电影海报插画',
  Cubism: '立体主义',
  'Carne Griffiths Style': '卡恩·格里菲斯风格',
  'Jackson Pollock Style': '杰克逊·波洛克风格',
  'Watercolor Painting': '水彩画',
  'Contemporary Oil Painting': '当代油画',
  'Pop Art': '波普艺术',
  'Thick Brushstrokes Painting': '厚笔触绘画',
  'Ashcan School Style': '垃圾箱画派风格',
  'Charles Tunnicliffe Style': '查尔斯·特尼克利夫风格',
  Cloisonnism: '掐丝珐琅画风',
  'Salvador Dalí Style': '达利风格',
  'Contemporary Gothic Art': '当代哥特艺术',
  'Norman Rockwell Style': '诺曼·罗克韦尔风格',
  'Edvard Munch Style': '蒙克风格',
  'Modern Klimt-Inspired Flat Collage Art': '现代克里姆特风平面拼贴',
  'Henri Matisse Style': '马蒂斯风格',
  'Andy Warhol Style': '沃霍尔风格',
  Impasto: '厚涂',
}

/* ------------------------------------------------------------------ *
 * 角色属性标签（官方 `character_presets_v2.json`，15 组 270 条）
 * ------------------------------------------------------------------ */

/**
 * 属性标签分组标题。键是 `item.group`（官方 `character_presets_v2.json` 里的原值）。
 */
export const TAG_GROUP_LABEL: Record<string, string> = {
  'Humanoid peoples': '人形种族',
  'Humanlike animal folk': '兽人',
  'Anthropomorphic species': '兽形种族',
  'Hybrid bodies': '混合体型',
  'Supernatural beings': '超自然存在',
  'Artificial and alien beings': '人工与外星生物',
}

/**
 * 标签**分类**名（`character_presets_v2.json` 的顶层键）。
 *
 * 官方只给 `races` 那一组设了二级分组（`item.group`），其余分类的
 * `group` 是空的，界面上的分组标题会退回到分类名（`skin_color` → "skin color"）。
 * 那些也全是英文，所以这一张表得补上。
 */
export const TAG_CATEGORY_LABEL: Record<string, string> = {
  races: '种族',
  'races ': '种族',
  skin_color: '肤色',
  'skin color': '肤色',
  body_type: '体型',
  'body type': '体型',
  breast_size: '胸型',
  'breast size': '胸型',
  face_shape: '脸型',
  'face shape': '脸型',
  face_details: '面部细节',
  'face details': '面部细节',
  hair_color: '发色',
  'hair color': '发色',
  hair_pattern: '发色图案',
  'hair pattern': '发色图案',
  hair_length: '发长',
  'hair length': '发长',
  hair_texture: '发质',
  'hair texture': '发质',
  hairstyles: '发型',
  hair_framing: '额发',
  'hair framing': '额发',
  eye_color: '瞳色',
  'eye color': '瞳色',
  eye_features: '眼部特征',
  'eye features': '眼部特征',
  details: '身体细节',
}

/**
 * 标签分组标题的中文名。
 *
 * 官方有两种来源：二级分组（`item.group`，如 "Humanoid peoples"）与
 * 分类名兜底（如 `skin_color`）。两张表都查一遍。
 */
export function tagGroupLabel(header: string): string {
  return TAG_GROUP_LABEL[header] ?? TAG_CATEGORY_LABEL[header] ?? header
}

/**
 * 标签 chip 上的中文名。
 *
 * 查表用 **`tag`**（`blue_eyes`）而不是 `label`（`Blue`）——
 * `tag` 才是写进 prompt 的内容，映射键必须对得上它。
 */
export function tagLabel(tag: string, label?: string): string {
  return TAG_LABEL[tag] ?? label ?? tag.split('_').join(' ')
}

/**
 * 属性标签中文名，键是 `item.tag`（270 条互不重复，已校验）。
 *
 * **键必须是 `tag` 而不是 `label`** —— `tag` 才是写进 prompt 的内容
 * （`blue_eyes`），`label` 只是给人看的短名。
 */
export const TAG_LABEL: Record<string, string> = {
  /* ---- 种族 ---- */
  human: '人类',
  elf: '精灵',
  'dark elf': '暗精灵',
  dwarf: '矮人',
  halfling: '半身人',
  gnome: '地精',
  orc: '兽人',
  goblin: '哥布林',
  ogre: '食人妖',
  troll: '巨怪',
  giant: '巨人',
  cyclops: '独眼巨人',
  catfolk: '猫族',
  foxfolk: '狐族',
  wolfkin: '狼族',
  dogfolk: '犬族',
  rabbitfolk: '兔族',
  horsefolk: '马族',
  deerfolk: '鹿族',
  goatfolk: '羊族',
  mousefolk: '鼠族',
  squirrelfolk: '松鼠族',
  'feline beastfolk': '猫兽人',
  'canine beastfolk': '犬兽人',
  bearfolk: '熊族',
  avian: '鸟族',
  lizardfolk: '蜥蜴族',
  dragonkin: '龙裔',
  kobold: '狗头人',
  sharkfolk: '鲨族',
  insectoid: '虫族',
  mothfolk: '蛾族',
  merfolk: '人鱼族',
  naga: '娜迦 / 拉米亚',
  centaur: '半人马',
  satyr: '萨堤尔 / 牧神',
  harpy: '鹰身女妖',
  arachne: '阿拉克涅（蜘蛛女妖）',
  minotaur: '牛头人',
  angel: '天使',
  demon: '恶魔',
  tiefling: '提夫林',
  oni: '鬼',
  vampire: '吸血鬼',
  fairy: '妖精',
  dryad: '树精',
  slimefolk: '史莱姆族',
  ghost: '幽灵',
  skeleton: '活体骷髅',
  zombie: '僵尸',
  djinn: '灯神',
  'living flame': '火元素',
  'water elemental': '水元素',
  'crystal being': '水晶生命体',
  android: '仿生人',
  robot: '机器人',
  cyborg: '改造人',
  'automaton doll': '人偶',
  golem: '石头魔像',
  'grey alien': '灰色外星人',
  'fungal being': '真菌生命体',

  /* ---- 肤色 ---- */
  'porcelain skin': '瓷白',
  'fair skin': '白皙',
  'light beige skin': '浅米色',
  'golden tan skin': '蜜色',
  'olive skin': '橄榄色',
  'warm brown skin': '暖棕',
  'deep brown skin': '深棕',
  'ebony skin': '乌木黑',
  'ivory skin': '象牙白',
  'ashen grey skin': '灰白',
  'slate blue skin': '石板蓝',
  'sky blue skin': '天蓝',
  'teal skin': '蓝绿',
  'jade green skin': '翡翠绿',
  'lavender skin': '薰衣草紫',
  'violet skin': '紫罗兰',
  'rose pink skin': '玫瑰粉',
  'crimson skin': '绯红',
  'pure white skin': '纯白',
  'charcoal black skin': '炭黑',

  /* ---- 体型 ---- */
  'slim build': '纤细',
  'average build': '普通',
  'lean athletic build': '精瘦健硕',
  'muscular build': '肌肉发达',
  'stocky build': '壮实',
  'full-bodied build': '丰满',
  'broad shoulders': '宽肩',
  'narrow shoulders': '窄肩',
  'wide hips': '宽胯',
  'narrow hips': '窄胯',
  'defined waist': '腰线分明',
  'straight waist': '平直腰',
  'long legs': '长腿',
  'short legs': '短腿',
  'long torso': '长躯干',
  'short torso': '短躯干',
  'tall stature': '高挑',
  'short stature': '矮小',
  'visible abdominal muscles': '腹肌分明',
  'soft abdomen': '柔软小腹',

  /* ---- 胸型 ---- */
  breasts: '胸部（通用）',
  large_breasts: '大胸',
  huge_breasts: '巨乳',
  medium_breasts: '中等胸型',
  small_breasts: '小胸',
  flat_chest: '平胸',
  'alternate_breast_size_(larger)': '备用胸型（更大）',
  mole_on_breast: '胸痣',

  /* ---- 脸型 ---- */
  'oval face': '鹅蛋脸',
  'round face': '圆脸',
  'square jaw': '方下巴',
  'heart-shaped face': '心形脸',
  'long narrow face': '长脸',
  'high cheekbones': '高颧骨',
  'full cheeks': '饱满面颊',
  'hollow cheeks': '凹陷面颊',
  'pointed chin': '尖下巴',
  'cleft chin': '带沟的下巴',
  'straight nose': '直鼻',
  'aquiline nose': '鹰钩鼻',
  'upturned nose': '上翘鼻',
  'broad nose': '宽鼻',
  'full lips': '丰唇',
  'thin lips': '薄唇',
  'thick eyebrows': '浓眉',
  'thin eyebrows': '细眉',
  'arched eyebrows': '弯眉',

  /* ---- 面部细节 ---- */
  'freckles across nose and cheeks': '面部雀斑',
  'beauty mark below one eye': '眼下泪痣',
  'beauty mark near the lips': '唇边美人痣',
  'scar across one cheek': '脸颊伤疤',
  'scar through one eyebrow': '眉上疤痕',
  'dark circles under the eyes': '黑眼圈',
  'facial birthmark': '面部胎记',
  'fine facial wrinkles': '细纹',
  'acne marks on the face': '痘痘印',
  'cheek dimples': '酒窝',
  'short facial stubble': '短胡茬',
  'trimmed beard': '修整胡须',
  'long beard': '长胡须',
  moustache: '八字胡',
  'subtle eyeliner': '淡眼线',
  'smoky eye makeup': '烟熏眼妆',
  'red lipstick': '红色唇膏',
  'black lipstick': '黑色唇膏',
  'geometric face paint': '几何面绘',

  /* ---- 发色 ---- */
  'black hair': '黑发',
  'brown hair': '棕发',
  'chestnut hair': '栗色头发',
  'auburn hair': '赤褐发',
  'copper hair': '铜色头发',
  'blonde hair': '金发',
  'platinum blonde hair': '铂金发',
  'white hair': '白发',
  'grey hair': '灰发',
  'silver hair': '银发',
  'pink hair': '粉发',
  'red hair': '红发',
  'orange hair': '橙发',
  'purple hair': '紫发',
  'blue hair': '蓝发',
  'teal hair': '蓝绿发',
  'green hair': '绿发',

  /* ---- 发色图案 ---- */
  'split-dyed hair with two color halves': '双色分区染',
  'hair with contrasting streaks': '对比色挑染',
  'hair with a gradual color transition toward the tips': '渐变发色',
  'hair with a contrasting inner layer': '内层撞色',
  'hair with rainbow-colored bands': '彩虹发带',

  /* ---- 发长 ---- */
  'shaved head': '剃光',
  'buzz cut hair': '寸头',
  'ear-length hair': '耳长',
  'chin-length hair': '齐下巴',
  'shoulder-length hair': '齐肩',
  'waist-length hair': '及腰',
  'floor-length hair': '及地',

  /* ---- 发质 ---- */
  'straight hair': '直发',
  'wavy hair': '波浪发',
  'curly hair': '卷发',
  'tightly coiled hair': '螺旋卷',
  'fluffy voluminous hair': '蓬松',
  'sleek hair': '顺滑',
  'tousled hair': '凌乱',

  /* ---- 发型 ---- */
  'bob haircut': '波波头',
  'pixie haircut': '精灵短发',
  'layered shag haircut': '层次碎盖',
  'undercut hairstyle': '侧铲',
  'mohawk hairstyle': '莫西干',
  'high ponytail': '高马尾',
  'low ponytail': '低马尾',
  'side ponytail': '侧马尾',
  'twin ponytails': '双马尾',
  'single hair braid': '单辫',
  'twin hair braids': '双辫',
  'crown braid hairstyle': '头顶编发',
  'cornrow braids': '贴头辫',
  'locs hairstyle': '脏辫',
  'single hair bun': '单丸子头',
  'double hair buns': '双丸子头',
  'half-up hairstyle': '半扎',
  'loose hair worn down': '披发',
  'spiral ringlet hair': '螺旋卷发',

  /* ---- 额发 ---- */
  'blunt bangs': '齐刘海',
  'curtain bangs': '空气刘海',
  'side-swept bangs': '斜刘海',
  'hair parted in the middle': '中分',
  'hair parted at the side': '侧分',
  'slicked-back hair': '大背头',
  'hair covering one eye': '遮眼刘海',
  'face-framing hair strands': '修饰脸型的发丝',
  'single upturned hair tuft': '一撮翘毛',

  /* ---- 瞳色 ---- */
  'blue eyes': '蓝眼',
  'brown eyes': '棕眼',
  'green eyes': '绿眼',
  'hazel eyes': '榛色眼',
  'amber eyes': '琥珀眼',
  'grey eyes': '灰眼',
  'black eyes': '黑眼',
  'violet eyes': '紫眼',
  'pink eyes': '粉眼',
  'red eyes': '红眼',
  'yellow eyes': '黄眼',
  'turquoise eyes': '绿松石眼',
  'orange eyes': '橙眼',
  'silver eyes': '银眼',

  /* ---- 眼部特征 ---- */
  'eyes with differently colored irises': '异色瞳',
  'irises with two distinct color sectors': '环形异色瞳',
  'almond-shaped eyes': '杏仁眼',
  'round eyes': '圆眼',
  'hooded eyelids': '深眼睑',
  'upturned outer eye corners': '上挑眼角',
  'downturned outer eye corners': '下垂眼角',
  'vertical slit pupils': '竖瞳',
  'horizontal slit pupils': '横瞳',
  'star-shaped pupils': '星星眼',
  'heart-shaped pupils': '爱心眼',
  'luminous irises': '发光瞳孔',
  'black sclerae': '黑色巩膜',
  'long eyelashes': '长睫毛',
  'short eyelashes': '短睫毛',

  /* ---- 身体细节 ---- */
  'geometric body tattoos': '几何纹身',
  'floral body tattoos': '花卉纹身',
  'full sleeve tattoo': '满臂纹身',
  'body scars': '身上疤痕',
  'body birthmark': '身体胎记',
  'freckles on shoulders and arms': '肩臂雀斑',
  'patches of depigmented skin': '白斑',
  'stretch marks': '妊娠纹',
  'visible veins': '可见血管',
  'tan lines': '晒痕',
  'geometric body paint': '几何人体彩绘',
  'luminous markings on the skin': '发光纹路',
  'metallic fingernail polish': '金属美甲',
  'black fingernails': '黑指甲',
  'elongated clawlike nails': '尖长指甲',
  'ear piercings': '耳洞',
  'nose piercing': '鼻环',
  'lip piercing': '唇钉',
  'prosthetic arm': '义肢手臂',
  'prosthetic leg': '义肢腿',
  'pointed ears': '尖耳',
  'animal ears': '兽耳',
  'curved horns': '弯角',
  'branched horns': '分叉角',
  'feathered wings': '羽翼',
  'batlike wings': '蝠翼',
  'long fur-covered tail': '毛茸长尾',
  'scaled tail': '鳞尾',
  'long canine fangs': '尖牙',
  'patches of scales on the skin': '鳞片',

  /* ---- 下面是 `character_tags.json`（属性选择器用的第二份官方库）----
   * 这份库按 hair_color / hairstyles / races / details 分组，tag 是下划线写法
   * （`long_hair`）。以前只翻了 `character_presets_v2.json`，于是选择器里
   * 发色 / 发型 / 种族那一排全显示英文。 */
  /* 发色 */
  black_hair: '黑发',
  brown_hair: '棕发',
  blonde_hair: '金发',
  grey_hair: '灰发',
  white_hair: '白发',
  pink_hair: '粉发',
  purple_hair: '紫发',
  blue_hair: '蓝发',
  green_hair: '绿发',
  red_hair: '红发',
  orange_hair: '橙发',
  multicolored_hair: '多色发',
  'two-tone_hair': '双色发',
  streaked_hair: '挑染',
  gradient_hair: '渐变发',
  colored_inner_hair: '内侧染色',
  dyed_hair: '染发',

  /* 发型 */
  long_hair: '长发',
  short_hair: '短发',
  very_long_hair: '超长发',
  medium_hair: '中长发',
  ponytail: '马尾',
  twintails: '双马尾',
  braid: '麻花辫',
  side_ponytail: '侧马尾',
  twin_braids: '双麻花辫',
  drill_hair: '钻头卷',
  ahoge: '呆毛',
  sidelocks: '鬓发',
  blunt_bangs: '齐刘海',
  parted_bangs: '中分刘海',
  bangs: '刘海',
  hair_bun: '丸子头',
  side_swept_bangs: '斜刘海',
  two_side_up: '两侧扎起',
  messy_hair: '乱发',
  floating_hair: '飘发',
  hair_over_one_eye: '遮单眼',
  drills: '螺旋卷',

  /* 种族特征 */
  animal_ears: '兽耳',
  cat_girl: '猫娘',
  fox_girl: '狐娘',
  rabbit_girl: '兔娘',
  horse_girl: '马娘',
  kemonomimi: '兽耳娘',
  furry: '兽人',
  furry_female: '女性兽人',
  demon_girl: '恶魔娘',
  demon_horns: '恶魔角',
  dragon_girl: '龙娘',
  dragon_horns: '龙角',
  wings: '翅膀',
  feathered_wings: '羽翼',
  pointy_ears: '尖耳',
  mermaid: '人鱼',
  rabbit_ears: '兔耳',
  horse_tail: '马尾巴',
  tail: '尾巴',

  /* 附加细节 */
  tattoo: '纹身',
  piercing: '穿孔',
  scar_on_body: '身上疤痕',
  scar_on_face: '脸上疤痕',
  veins: '血管',
  nail_polish: '指甲油',
  black_nails: '黑指甲',
  tan_lines: '晒痕',
  mole: '痣',
  birthmark: '胎记',
  freckles: '雀斑',
  body_paint: '身体彩绘',
  smudged_makeup: '花了的妆',
}

/* ------------------------------------------------------------------ *
 * 姿势库（官方 `PoseLibrary/`，13 个分类 191 个姿势）
 * ------------------------------------------------------------------ */

/**
 * 姿势分类中文名。键是目录名（`Ballet` / `Close Up` / …）。
 *
 * `kneeling`（小写）也在表里 —— 官方两个仓库各写各的，同一个分类有
 * `Kneeling` 和 `kneeling` 两种写法，后端只做首字母去重，条目本身还带着小写。
 */
export const POSE_CATEGORY_LABEL: Record<string, string> = {
  Ballet: '芭蕾',
  'Close Up': '特写',
  Crawling: '爬行',
  Crouching: '蹲伏',
  Fighting: '战斗',
  Gymnastics: '体操',
  Kneeling: '跪姿',
  kneeling: '跪姿',
  Laying: '躺卧',
  Rest: '休息',
  Sitting: '坐姿',
  Standing: '站立',
  Uncategorized: '未分类',
  Yoga: '瑜伽',
}

/**
 * 姿势名中文名，键是文件名（去扩展名后的原样）。
 *
 * 官方文件名有些怪癖，这里**一字不改地当键**：
 * `CloseUp  B`（两个空格）、`Sit on chair С`（结尾是西里尔字母 Es 不是 C）、
 * `Legs Crosed`（官方自己拼错了）、`Dancing_on_knees`（下划线）。
 * 改一个字符就匹配不上，用户点的姿势会变成另一个。
 */
export const POSE_NAME_LABEL: Record<string, string> = {
  /* 芭蕾 */
  Arabesque: '迎风展翅式',
  Dancing_on_knees: '跪姿起舞',
  Splits_1: '劈叉 1',

  /* 特写 */
  'Peace Sign - Close Up': '比耶手势 - 特写',

  /* 爬行 */
  Crawling: '爬行',
  'Crawling - Rear View': '爬行 - 背面',
  'Crawling rear': '爬行 - 背面',

  /* 蹲伏 */
  'Crouching - Side View': '蹲伏 - 侧面',

  /* 战斗 */
  'Boxing Stance': '拳击预备式',
  'Fighting - Right Jab': '战斗 - 右直拳',
  'Fighting A': '战斗 A',
  'Fighting B': '战斗 B',
  'Fighting C': '战斗 C',
  'Fighting D': '战斗 D',
  'Fighting Straight Punch': '战斗 - 直拳',

  /* 体操 */
  Handstand: '手倒立',
  'Middle Splits': '正劈叉',

  /* 跪姿 */
  'Kneeling - Side View': '跪姿 - 侧面',
  'Kneeling on One Knee': '单膝跪地',
  Seiza: '正坐',
  Kneeling: '跪姿',
  'Kneeling Forward on Hands': '跪姿前撑',
  'Kneeling Leaning Back': '跪姿后仰',
  'Kneeling Legs Apart': '跪姿 - 双腿分开',
  'Kneeling Legs Spread': '跪姿 - 双腿展开',
  'Kneeling overhead': '跪姿 - 俯视',
  'Kneeling side': '跪姿 - 侧面',

  /* 躺卧 */
  'Fetal Position - Overhead': '胎儿蜷缩 - 俯视',
  'Fetal Position - Rear View': '胎儿蜷缩 - 背面',
  'Fetal overhead': '胎儿蜷缩 - 俯视',
  'Fetal rear': '胎儿蜷缩 - 背面',

  /* 休息 */
  'CloseUp  B': '特写 B',
  'CloseUp  C': '特写 C',
  'CloseUp  D': '特写 D',
  'CloseUp A': '特写 A',
  'Lying A': '躺卧 A',
  'Lying B': '躺卧 B',
  'Lying C': '躺卧 C',
  'Lying D': '躺卧 D',
  'Sit on chair A': '坐椅 A',
  'Sit on chair B': '坐椅 B',
  'Sit on chair D': '坐椅 D',
  'Sit on chair С': '坐椅 C',
  'Standing A': '站立 A',
  'Standing B': '站立 B',
  'Standing C': '站立 C',
  'Standing D': '站立 D',

  /* 坐姿 */
  'Close Up - Overhead View': '特写 - 俯视',
  'Leaning Back': '后仰',
  Sitting: '坐姿',
  'Sitting - Rear View': '坐姿 - 背面',
  'Sitting - Side View': '坐姿 - 侧面',
  'Sitting Cross Legged': '盘腿坐',
  'Sitting legs crossed': '坐姿 - 双腿交叉',
  'sitting up on side': '侧坐',
  'Seated on Floor': '坐地',

  /* 站立 */
  'Arms Behind Back - Side View': '双手背后 - 侧面',
  'Arms Crossed': '双臂交叉',
  'Arms Extended Backwards - Side View': '双臂后展 - 侧面',
  'Arms Extended Forward - Side View': '双臂前伸 - 侧面',
  'Arms Outstretched': '双臂张开',
  'Arms Raised': '双臂抬起',
  'Arms Up': '双臂上举',
  Contrapposto: '对立式平衡',
  'Contrapposto - Hand on Hip': '对立式平衡 - 单手叉腰',
  'Contrapposto - Rear View': '对立式平衡 - 背面',
  'Feet Apart - Side View': '双脚分开 - 侧面',
  'Feet Apart': '双脚分开',
  'Feet Apart profile': '双脚分开 - 侧面',
  'feet apart rear': '双脚分开 - 背面',
  'Hands on hips - Rear View': '双手叉腰 - 背面',
  'Leaning Back - Side View': '后仰 - 侧面',
  'Leaning Forward': '前倾',
  'Leaning Forward - Side View': '倾 - 侧面',
  'Leaning Back legs spread': '后仰 - 双腿分开',
  'One Arm Raised': '单臂抬起',
  'One Hand on Hip': '单手叉腰',
  'Slight Lean Back': '微微后仰',
  Standing: '站立',
  'Standing - Hands Behind Back': '站立 - 双手背后',
  'Standing - Leaning Forward': '站立 - 前倾',
  'Standing - Rear View': '站立 - 背面',
  'Standing - Side View': '站立 - 侧面',
  Stepping: '迈步',
  'Touching Toes - Side View': '摸脚尖 - 侧面',
  'Waving - Overhead View': '挥手 - 俯视',
  'Weight on One Leg - Side View': '单腿重心 - 侧面',
  'Bent over toe touch profile': '弯腰触脚尖 - 侧面',
  'forward step': '前迈步',
  Jumping: '跳跃',
  Kicking: '踢腿',
  'knee raised profile': '抬膝 - 侧面',
  'raised foot': '抬腿',
  'Prone': '俯卧',
  'Prone Feet Up profile': '俯卧 - 双脚上翘 - 侧面',
  'Prone Legs Up Side': '俯卧 - 双腿上抬 - 侧面',
  'Prone on Knees side': '跪伏 - 侧面',
  'Legs Crosed': '双腿交叉',
  'Laying Back Knees Spread Apart': '仰卧 - 双膝分开',
  'Laying Back Knees Up Together': '仰卧 - 双膝并拢',
  'Laying Back Legs Akimbo': '仰卧 - 叉开双腿',
  'Laying Back Legs Crossed': '仰卧 - 双腿交叉',
  'Laying Back Legs Spread': '仰卧 - 双腿展开',
  'Laying Back on Elbows': '仰卧撑肘',
  'Laying on Back': '仰卧',
  Squatting: '深蹲',
  'Squatting Forward on Hands': '深蹲前撑',
  'Squatting Leaning Back': '深蹲后仰',
  'Squatting Leg Extended': '深蹲 - 单腿伸展',
  'Squatting Legs Spread': '深蹲 - 双腿分开',
  'Squatting overhead': '深蹲 - 俯视',
  'Squatting rear': '深蹲 - 背面',
  'Squatting side': '深蹲 - 侧面',
  'Standing Hand on Hip': '站立 - 单手叉腰',
  'Standing Leaning Back': '站立 - 后仰',
  'Standing Leaning Forward': '站立 - 前倾',
  'Standing One Hand on Hip 2': '站立 - 单手叉腰 2',
  'Standing One Leg Up': '站立 - 抬单腿',
  'Standing profile': '站立 - 侧面',
  'standing profile right': '站立 - 右侧',
  'standing rear': '站立 - 背面',
  'Walking profile': '行走 - 侧面',

  /* 瑜伽 */
  'Downward Dog': '下犬式',
  Warrior: '战士式',

  /* 无名 */
  1: '姿势 1',
}

/**
 * 姿势标签中文名（官方库 106 条）。
 *
 * `front  view`（两个空格）和 `front view`（一个空格）**两个都在表里** ——
 * 官方文件里两种写法都出现过，少一个就有一条标签露英文。
 */
export const POSE_TAG_LABEL: Record<string, string> = {
  Fighting: '战斗',
  'Leaning back': '后仰',
  Sitting: '坐姿',
  arabesque: '迎风展翅式',
  'arms behind back': '双手背后',
  'arms crossed': '双臂交叉',
  'arms extended': '双臂伸展',
  'arms extended backwards': '双臂后展',
  'arms outstretched': '双臂张开',
  'arms raised': '双臂抬起',
  'arms up': '双臂上举',
  ballet: '芭蕾',
  'bent over': '弯腰',
  boxing: '拳击',
  'boxing stance': '拳击预备式',
  calm: '平静',
  'close up': '特写',
  closeup: '特写',
  contrapposto: '对立式平衡',
  crawling: '爬行',
  'cross legged': '盘腿',
  crouching: '蹲伏',
  dancing: '舞蹈',
  'downward dog': '下犬式',
  'elbows propped': '撑肘',
  'feet apart': '双脚分开',
  'feet together': '双脚并拢',
  'feet up': '双脚上抬',
  fetal: '胎儿蜷缩',
  'fetal position': '胎儿蜷缩',
  fighting: '战斗',
  'fighting stance': '战斗预备式',
  'forward on hands': '前撑',
  'forward step': '前迈步',
  'front  view': '正面',
  'front view': '正面',
  'full body': '全身',
  gym: '体操',
  'hand on hip': '单手叉腰',
  'hands behind back': '双手背后',
  'hands on hips': '双手叉腰',
  hands_up: '双手上举',
  handstand: '手倒立',
  'head back': '头后仰',
  jab: '直拳',
  jumping: '跳跃',
  kicking: '踢腿',
  'knee raised': '抬膝',
  kneeling: '跪姿',
  knees: '膝盖',
  'knees apart': '双膝分开',
  'knees bent': '屈膝',
  'knees together': '双膝并拢',
  'knees up': '双膝上抬',
  laying: '躺卧',
  'laying back': '仰卧',
  'laying on back': '仰卧',
  leaning: '倾斜',
  'leaning back': '后仰',
  'leaning back on hands': '后撑',
  'leaning forward': '前倾',
  'leaning on side': '侧倾',
  'leg extended': '单腿伸展',
  'legs akimbo': '双腿叉开',
  'legs apart': '双腿分开',
  'legs crossed': '双腿交叉',
  'legs open': '双腿张开',
  'legs spread': '双腿展开',
  'legs up': '双腿上抬',
  'looking up': '向上看',
  lying: '躺卧',
  'middle splits': '正劈叉',
  'on knees': '跪地',
  'one arm raised': '单臂抬起',
  'one hand on hip': '单手叉腰',
  'one knee': '单膝',
  'one leg': '单腿',
  overhead: '俯视',
  'overhead view': '俯视',
  peace: '比耶',
  posing: '摆姿势',
  'profile view': '侧面',
  prone: '俯卧',
  'raised foot': '抬脚',
  'rear view': '背面',
  relaxed: '放松',
  rest: '休息',
  'right facing': '朝右',
  'seated on floor': '坐地',
  seiza: '正坐',
  'side view': '侧面',
  sit: '坐',
  sitting: '坐姿',
  'slight lean back': '微微后仰',
  splits: '劈叉',
  squatting: '深蹲',
  standing: '站立',
  stepping: '迈步',
  'straight punch': '直拳',
  tiptoes: '踮脚',
  'touching toes': '摸脚尖',
  walking: '行走',
  warrior: '战士式',
  waving: '挥手',
  'weight on one leg': '单腿重心',
  yoga: '瑜伽',
}




