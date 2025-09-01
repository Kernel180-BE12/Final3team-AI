<!--
METADATA:
  file_type: "content_guide"
  template_specs: true
  severity: "low"
  content_type: "info"
  compliance_level: "optional"
  keywords: ["알림톡", "템플릿"]
  main_topic: "알림톡 가이드"
  purpose: "guide"
  business_impact: "medium"
  tags: ["알림톡", "가이드"]
  source_file: "cleaned_content-guide.md"
  chunk_id: 1
  header: ""
  content_length: 71
-->

# 제작 가이드

{% hint style="info" %}
알림톡 템플릿 제작가이드를 알아보겠습니다.
{% endhint %}

<!--
METADATA:
  file_type: "content_guide"
  template_specs: true
  severity: "low"
  content_type: "general"
  compliance_level: "optional"
  keywords: []
  main_topic: "알림톡 가이드"
  purpose: "guide"
  business_impact: "medium"
  tags: ["알림톡", "가이드"]
  source_file: "cleaned_content-guide.md"
  chunk_id: 2
  header: "## **1. 메시지 유형**"
  content_length: 16
-->

## **1. 메시지 유형**

<!--
METADATA:
  file_type: "content_guide"
  template_specs: true
  section: "1-1. 기본형"
  subsection: "1-1. 기본형"
  severity: "low"
  content_type: "general"
  compliance_level: "mandatory"
  keywords: ["발송"]
  main_topic: "알림톡 가이드"
  purpose: "guide"
  business_impact: "medium"
  tags: ["알림톡", "가이드"]
  source_file: "cleaned_content-guide.md"
  chunk_id: 3
  header: "### 1-1. 기본형"
  content_length: 118
-->

### 1-1. 기본형


고객에게 반드시 전달되어야하는 정보를 발송할 수 있습니다.

* 한/영 구분없이 1,000자까지 입력 가능합니다.&#x20;
* 개인화된 텍스트 영역은 #{변수}로 작성할 수 있습니다.

<!--
METADATA:
  file_type: "content_guide"
  template_specs: true
  section: "1-2. 부가 정보형"
  subsection: "1-2. 부가 정보형"
  severity: "high"
  content_type: "example"
  compliance_level: "mandatory"
  keywords: ["알림톡", "정보성", "광고성", "발송"]
  main_topic: "알림톡 가이드"
  purpose: "guide"
  business_impact: "medium"
  tags: ["알림톡", "가이드"]
  source_file: "cleaned_content-guide.md"
  chunk_id: 4
  header: "### 1-2. 부가 정보형"
  content_length: 1231
-->

### 1-2. 부가 정보형


(1) 고객에게 고정적인 부가 정보에 대한 안내가 지속적으로 필요한 경우 사용할 수 있습니다.

* 알림톡 메시지의 본문 하단에 노출되며, 이용안내 등 보조적인 정보메시지에 대해 안내가 가능합니다.
* 부가정보는 최대 500자로 변수 사용이 불가능하며, URL은 포함 가능합니다.
* 광고성 요소와 동시에 사용 가능하며, 부가정보 와 광고성 문구 총 500자로 제한되고, 본문과 합쳐 총 1,000자를 넘을 수 없습니다.
* 해당 영역은 본문과 버튼 사이 Description영역에 노출되며, 본문 대비 사이즈 -1pt, 컬러차가 자동 적용됩니다.

![](https://234308570-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2F-MVZVmVOd-5LtENUPqdq%2Fuploads%2Fm1v0RobW03acxEl9GMB4%2Fimage.png?alt=media\&token=0c6898e2-6566-4193-b96e-46fd487d3669)

* 사용 예시

(2) 매월 정기적으로 발송되는 메시지 하단에 광고 메시지가 링크로 포함 된 경우 사용할 수 있습니다.

* 매월 1회 정기적으로 발송되는 정보성 메시지에는 “카드사의 청구서”, “유료 멤버쉽의 월 갱신 안내”, “매월 신용정보 변동 종합 안내” 등이 해당됩니다.
* 광고 메시지는 변수 사용과 url 포함이 불가능하며, 페이지 이동은 WL(웹링크) 버튼 타입이 사용됩니다. (본문과 합쳐 총 1,000자를 넘을 수 없음)
* 이벤트에 대한 간단한 안내 외의 ‘아래 버튼을 누르시면 이벤트(또는 광고) 페이지로 이동합니다.’를 필수로 안내해야 합니다.
* 해당 영역은 본문과 버튼 사이 Description영역에 노출되며, 본문 대비 사이드 -1pt, 컬러차가 자동 적용됩니다.

{% hint style="success" %}
\[OO카드] 카카오톡 명세서\
\#{고객명} 회원님! #{결제월}월 결제금액에 대한 명세서입니다.\
■ 당월 결제금액\
\#{청구금액}원\
■ 모바일 명세서로 보기\
\#{url}\

<mark style="color:blue;">카카오톡 명세서 수신 고객 대상으로 OO이벤트가 진행 중입니다.</mark>\ <mark style="color:blue;">아래 버튼을 누르시면 이벤트 페이지로 이동합니다.</mark>\
\#{이벤트 페이지 이동 버튼}
{% endhint %}

* 사용 예시

<!--
METADATA:
  file_type: "content_guide"
  template_specs: true
  section: "1-3. 채널추가형"
  subsection: "1-3. 채널추가형"
  severity: "low"
  content_type: "example"
  compliance_level: "optional"
  keywords: ["광고성", "승인", "채널"]
  main_topic: "알림톡 가이드"
  purpose: "guide"
  business_impact: "medium"
  tags: ["알림톡", "가이드"]
  source_file: "cleaned_content-guide.md"
  chunk_id: 5
  header: "### 1-3. 채널추가형"
  content_length: 608
-->

### 1-3. 채널추가형


비광고성 메시지 하단에 카카오톡 채널 추가가 포함 된 경우 사용할 수 있습니다.

* 안내 멘트는 최대 80자로 변수 사용과 url 포함이 불가능합니다. (본문과 합쳐 총 1,000자를 넘을 수 없음, 부가정보형과 동시 사용 가능)
* 카카오톡 채널추가의 경우, 카카오톡 채널명을 제외한 영역은 아래 문구만 사용해야 하며, 수정이 불가능합니다.     \
  -채널 추가하고 이 채널의 마케팅 메시지 등을 카카오톡으로 받기
* 카카오톡 채널 추가는 AC(채널추가) 버튼 타입을 사용하며, 기타 수신동의는 WL(웹링크) 버튼 타입 사용을 사용합니다.
* 해당 영역은 본문과 버튼 사이 Description영역에 노출되며, 본문 대비 사이드 -1pt, 컬러차가 자동 적용됩니다.

{% hint style="success" %}
\[OO카드]\
\#{고객명} OO#{카드번호4자리}승인 \
\#{승인금액}원 #{할부개월}개월\
\#{거래일자} #{거래시각} \
\#{가맹점명}\

<mark style="color:blue;">채널 추가하고 이 채널의 마케팅 메시지 등을 카카오톡으로 받기</mark>\
\#{카카오톡 채널 추가 버튼}
{% endhint %}

<!--
METADATA:
  file_type: "content_guide"
  template_specs: true
  section: "1-4. 복합형"
  subsection: "1-4. 복합형"
  severity: "low"
  content_type: "general"
  compliance_level: "optional"
  keywords: ["채널"]
  main_topic: "알림톡 가이드"
  purpose: "guide"
  business_impact: "medium"
  tags: ["알림톡", "가이드"]
  source_file: "cleaned_content-guide.md"
  chunk_id: 6
  header: "### 1-4. 복합형"
  content_length: 61
-->

### 1-4. 복합형


(1) 부가정보형과 채널추가형을 동시에 사용하는 경우 복합형을 사용할 수 있습니다.

<!--
METADATA:
  file_type: "content_guide"
  template_specs: true
  section: "2. 강조 유형"
  severity: "low"
  content_type: "general"
  compliance_level: "optional"
  keywords: []
  main_topic: "알림톡 가이드"
  purpose: "guide"
  business_impact: "medium"
  tags: ["알림톡", "가이드"]
  source_file: "cleaned_content-guide.md"
  chunk_id: 7
  header: "## 2. 강조 유형"
  content_length: 11
-->

## 2. 강조 유형

<!--
METADATA:
  file_type: "content_guide"
  template_specs: true
  section: "2-1. 이미지형"
  subsection: "2-1. 이미지형"
  severity: "high"
  content_type: "general"
  compliance_level: "mandatory"
  keywords: ["알림톡", "정보성", "광고성", "템플릿", "발송"]
  main_topic: "알림톡 가이드"
  purpose: "guide"
  business_impact: "medium"
  tags: ["알림톡", "가이드"]
  source_file: "cleaned_content-guide.md"
  chunk_id: 8
  header: "### 2-1. 이미지형"
  content_length: 1954
-->

### 2-1. 이미지형


메시지가 포맷화된 정보성 메시지를 안내하기 위한 용도로 이미지형 알림톡을 사용할 수 있습니다.

**1) 이미지 알림톡 서비스 기준**

* 이미지 알림톡은 정보성 메시지로 수신자에게 반드시 전달되어야 하는 정보만 발송이 가능합니다. 따라서, 광고성 메시지는 발송이 불가능하며, 이미지에도 광고성 내용은 포함 될 수 없습니다.
* 알림톡에 포함될 수 있는 이미지는 발송 주체 또는 발송 주체의 톡 채널 프로필을 대표하는 이미지, 메시지의 목적을 표현하는 이미지로 제한하며, 본문과 관련성이 떨어지거나 반드시 전달되어야 하는 내용은 포함 될 수 없습니다.
* 강조표기형과 동시에 사용할 수 없으며, 카카오톡 8.7.5 버전(안드로이드, iOS 공통) 이상에서만 노출됩니다.
* 이미지 알림톡은 템플릿 당 하나의 고정된 이미지만 사용 가능하며, 이미지 제작 가이드를 준수하여야 합니다.

**2) 고정형 이미지 알림톡 제작 가이드**

&#x20;   **(1)  공통 가이드**

* 템플릿 당 하나의 고정 이미지만 사용 가능
* 권장 사이즈/파일형식/파일 용량 : 800×400px /JPG, PNG / 최대 500KB
* 가로:세로 비율이 2:1 아닐 경우, 가로 500px, 세로 250px 이하일 경우 업로드 불가&#x20;
* 이미지 클릭 시 링크 연결 불가

&#x20;   **(2) 로고형 이미지**

&#x20;  해당 채널의 가로형 로고를 활용해 메시지의 주목도를 높이고 브랜딩을 강화할 수 있는 유형

* 이미지 배경: #F9F9F9 만 사용 가능
* 이미지 배치: 센터 가이드라인 내 로고 로고 배치 &#x20;
* 사용 예시&#x20;

![](https://234308570-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2F-MVZVmVOd-5LtENUPqdq%2Fuploads%2FwqubusDvJRYjcmQEF6tm%2Fimage.png?alt=media\&token=2e4cdbc8-b331-4298-9580-037ea8be97c1)

&#x20;   **(3) 텍스트 혼합형 이미지**

&#x20;   주요 내용을 텍스트로 표현, 내용을 보조하는 아이콘이나 오브젝트 이미지를 활용하는 유형

* 이미지 배경: #F9F9F9 만 사용 가능
* 이미지 배치: 좌상단에 주요 내용 텍스트, 우하단에 아이콘 또는 오브젝트 삽입
* 텍스트 기준: 최대 2줄 / 20자 노출을 지향
* 사용 예시&#x20;

![](https://234308570-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2F-MVZVmVOd-5LtENUPqdq%2Fuploads%2F8wJktqduZrFcVdunWnjf%2Fimage.png?alt=media\&token=76f6ec89-29df-4611-b5c4-b2451b7a7028)

&#x20;   **(4)  아이콘형 이미지**

&#x20;  메시지 내용의 상태를 아이콘으로 표현하고 간략한 텍스트로 보조하는 유형

* 이미지 배경: #F9F9F9 만 사용 가능
* 이미지 배치: 센터 가이드라인 내 아이콘 및 텍스트 배치
* 텍스트 기준: 최대 1줄 / 최소 2자\~최대 9자 이하의 텍스트 노출
* 사용예시

![](https://234308570-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2F-MVZVmVOd-5LtENUPqdq%2Fuploads%2FzUA71kwVCzlwmc5N4QIi%2Fimage.png?alt=media\&token=9b89270c-ceab-4d78-9e13-637eaa0946dd)

{% content-ref url="content-guide/image" %}
[image](content-guide/image)
{% endcontent-ref %}

<!--
METADATA:
  file_type: "content_guide"
  template_specs: true
  section: "2-2. 강조표기형"
  subsection: "2-2. 강조표기형"
  severity: "medium"
  content_type: "general"
  compliance_level: "recommended"
  keywords: ["알림톡", "발송"]
  main_topic: "알림톡 가이드"
  purpose: "guide"
  business_impact: "medium"
  tags: ["알림톡", "가이드"]
  source_file: "cleaned_content-guide.md"
  chunk_id: 9
  header: "### 2-2. 강조표기형"
  content_length: 501
-->

### 2-2. 강조표기형


알림톡 내용에서 강조가 필요한 내용을 말풍선 상단 영역에 강조하여 표현할 수 있습니다.

**1) 타이틀 가이드 (말줄임 처리가 되지 않는 길이로 발송 권장)**

&#x20; **(1) Title**: 본문의 내용 중, 고객에게 강조하여 표현할 내용

&#x20;       a. 안드로이드: 최대 2줄 23자(24자부터 말줄임 처리)

&#x20;       b. iOS: 최대 2줄 27자(28자부터 말줄임 처리)

&#x20;  **(2) Subtitle:** Title에 어떤 내용이 들어가는지에 대한 부연 설명

&#x20;       a. 안드로이드: 최대 18자(19자부터 말줄임 처리)

&#x20;       b. iOS: 최대 21자(22자부터 말줄임 처리)

**2) 적용 조건**

* Title과 Subtitle은 함께 등록되어야 하며, 각각 단독으로 노출될 수 없음
* &#x20;Subtitle은 변수 등록 불가

**3) 사용 예시**

<!--
METADATA:
  file_type: "content_guide"
  template_specs: true
  section: "2-3. 아이템리스트형"
  subsection: "2-3. 아이템리스트형"
  severity: "high"
  content_type: "general"
  compliance_level: "mandatory"
  keywords: ["알림톡", "정보성", "광고성", "템플릿", "발송"]
  main_topic: "알림톡 가이드"
  purpose: "guide"
  business_impact: "medium"
  tags: ["알림톡", "가이드"]
  source_file: "cleaned_content-guide.md"
  chunk_id: 10
  header: "### 2-3. 아이템리스트형"
  content_length: 1807
-->

### 2-3. 아이템리스트형


메시지가 구조화된 정보성 메시지를 안내하기 위한 용도로 사용할 수 있습니다.

**1) 아이템리스트 서비스 기준**

* 알림톡 아이템리스트는 정보성 메시지로 수신자에게 반드시 전달되어야 하는 목록화된 정보에 한합니다. 따라서, 광고성 메시지는 발송이 불가능하며, 이미지와 목록화된 정보에도 광고성 내용은 포함 될 수 없습니다.
* 알림톡 아이템리스트에 포함될 수 있는 이미지는 발송 주체 또는 발송 주체의 톡 채널 프로필을 대표하는 이미지, 메시지의 목적을 표현하는 이미지로 제한하며, 본문과 관련성이 떨어지거나 반드시 전달되어야 하는 내용은 포함 될 수 없습니다.
* 강조표기형과 동시에 사용할 수 없으며, 카카오톡 9.3.5 버전(안드로이드, iOS 공통) 이상에서만 노출됩니다.
* 알림톡 아이템리스트는 템플릿 당 영역별 하나의 고정된 이미지만 사용 가능하며, 이미지 제작 가이드를 준수하여야 합니다.
* 보안템플릿 설정이 불가합니다.

**2) 아이템리스트형 제작가이드**

&#x20;     **(1) 아이템리스트**

* 아이템리스트는 최소 2개 \~ 최대 10개까지 등록 가능합니다.
* 아이템리스트 아이템명은 변수 사용이 불가하며, 디스크립션은 변수 사용이 가능합니다.
* 아이템 요약정보는 숫자와 통화기호, 통화단위 및 일부 특수기호(, .) 만 입력 가능합니다.
* 아이템 요약정보를 사용하면 리스트 항목들이 우측 정렬됩니다. (하단 사용 예시 4,5번 이미지) &#x20;
* 아이템명은 최대 6자, 디스트립션은 최대 23자 입력 가능하며, 2줄 초과 시 말줄임 처리됩니다.

&#x20;     **(2) 헤더**

* 최대 16자 입력가능합니다.&#x20;
* 변수 입력 가능합니다.

&#x20;     **(3) 아이템 하이라이트 텍스트, 디스크립션**

<table><thead><tr><th width="194.33333333333331" align="center">구분</th><th align="center">썸네일 없을 경우</th><th align="center">썸네일 있을 경우</th></tr></thead><tbody><tr><td align="center">텍스트</td><td align="center"><p>최대 30자(2줄) </p><p>1줄은 15자 입력가능</p></td><td align="center"><p>최대 21자(2줄) </p><p>1줄은 10자 입력가능</p><p>2줄 초과 시 말줄임 처리</p></td></tr><tr><td align="center">디스크립션</td><td align="center">최대 19자(1줄)</td><td align="center"><p>최대 13자(1줄)</p><p>1줄 초과 시 말줄임 처리</p></td></tr></tbody></table>

* 아이템 하이라이트 텍스트, 디스크립션은 변수 사용이 가능합니다.

&#x20;   **(4) 아이템리스트 하이라이트 썸네일이미지**

* 알림톡 텍스트 또는 아이템 하이라이트 영역과 연관된 이미지로 구성 (ex. 아이콘, 아이템이미지)
* 하이라이트 썸네일 이미지는 기존 이미지 가이드를 따르나 배경색은 자유 지정 가능
* 제한 사이즈 : 가로 108px 이상, 가로:세로 비율 1:1
* 파일형식 및 크기 : jpg, png / 최대 500KB

&#x20;   **(5) 아이템리스트 이미지**

* &#x20;권장 사이드 : 800px\*400px (jpg, png / 최대 500KB)
* 가로:세로 비율이 2:1 아닐 경우, 가로 500px 세로 250px 이하일 경우 업로드 불가

&#x20;   **(6) 사용 예시**

<!--
METADATA:
  file_type: "content_guide"
  template_specs: true
  section: "2-4. 메시지 내 서체 스타일 적용"
  subsection: "2-4. 메시지 내 서체 스타일 적용"
  severity: "low"
  content_type: "general"
  compliance_level: "optional"
  keywords: ["알림톡", "템플릿", "발송"]
  main_topic: "알림톡 가이드"
  purpose: "guide"
  business_impact: "medium"
  tags: ["알림톡", "가이드"]
  source_file: "cleaned_content-guide.md"
  chunk_id: 11
  header: "### 2-4. 메시지 내 서체 스타일 적용"
  content_length: 848
-->

### 2-4. 메시지 내 서체 스타일 적용


알림톡 내용에서 서체 스타일을 적용하여 표현할 수 있습니다.&#x20;

1\) 취소선

* 아이템 하이라이트 영역 내 타이틀, 강조표기형 내 타이틀 영역에서 취소선을 적용하고 싶을 경우 사용할 수 있습니다.
* 템플릿의 등록/수정 작업 없이도 알림톡 발송 시 허용된 타이틀 영역 마지막에 \\\s을 입력하면 취소선으로 인식하여 적용됩니다. 발송 시 말풍선 내 취소선 적용 플래그(\\\s)는 삭제되어 보여집니다.&#x20;
  * 예) "title": "87,000원\\\s”
* 허용된 영역 외 혹은 허용되지 않은 위치에 취소선을 적용하여 발송 요청 시 취소선 적용은 되지 않으며, 템플릿의 내용과 달라 발송 실패 처리 되거나 \\\s가 그대로 노출 될 수 있으니 사용 시 유의해주시기 바랍니다.&#x20;
* 카카오톡 10.0.0버전 (안드로이드, iOS 공통) 이상에서만 취소선이 적용됩니다.&#x20;

2\) 적용 필드 내 글자수 정책

<table><thead><tr><th width="224.33333333333331">구분</th><th width="244">템플릿 글자수 정책</th><th>발송 요청 시 글자수 정책</th></tr></thead><tbody><tr><td>아이템 하이라이트 > 타이틀</td><td>최대 30자 (썸네일 포함 시 21자)</td><td><p>취소선 플래그 제외 30자 </p><p>(썸네일 포함 시 플래그 제외 21자)</p></td></tr><tr><td>강조표기형 > 타이틀</td><td>최대 50자</td><td>취소선 플래그 제외 50자</td></tr></tbody></table>

3\) 사용 예시

<!--
METADATA:
  file_type: "content_guide"
  template_specs: true
  section: "3.미리보기 메시지"
  severity: "low"
  content_type: "info"
  compliance_level: "optional"
  keywords: ["알림톡", "템플릿", "심사", "발송", "채널"]
  main_topic: "알림톡 가이드"
  purpose: "guide"
  business_impact: "medium"
  tags: ["알림톡", "가이드"]
  source_file: "cleaned_content-guide.md"
  chunk_id: 12
  header: "## 3.미리보기 메시지"
  content_length: 355
-->

## 3.미리보기 메시지


알림톡 채팅방 리스트와 푸 메시지에 노출되는 문구 설정을 지원하는 기능으로 발송 목적을 효율적으로 노출시킬 수 있습니다.

* 한/영 구분없이 40자까지 입력 가능합니다.
* 메시지 내 변수 작성이 불가합니다.
* 수신자의 환경에 따라 전체 문구가 노출되지 않을 수 있습니다.
* 보안 템플릿은 미리보기 메시지 설정이 불가합니다.
* 심사 시 템플릿 설정값이 ‘보안템플릿’으로 변경될 수 있으며 변경 템플릿은 미리보기가 노출되지 않습니다.

{% hint style="info" %}
다른 문의사항이 있으실까요? 카카오톡 채널 혹은 고객센터를 통해 문의하실 수 있습니다.&#x20;
{% endhint %}